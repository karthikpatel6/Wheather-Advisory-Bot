"""
graph.py — LangGraph definition for the Weather Advisory Bot.

Graph shape:
    entry
      → resolve_location     (mostly-deterministic; regex-first, LLM-assist fallback)
      → fetch_weather         (deterministic; Open-Meteo API)
      → filter_numeric_sops   (pure Python; no LLM)
      → llm_select_sop        (single structured LLM call)
      → compose_reply         (LLM + placeholder substitution)

Failure branches (no LLM improvisation):
    honest_failure  — bad location or weather API failure
    no_guidance     — no SOP matched

State is persisted per thread_id via MemorySaver.
"""

from __future__ import annotations

import json
import logging
import operator as op_module
import os
import re
from datetime import datetime, timezone
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv(override=True)
from langchain_core.messages import HumanMessage, SystemMessage
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from policy_store import PolicyStore
from weather import LocationNotFoundError, WeatherFetchError, fetch_weather, geocode

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons — loaded once at import time
# ---------------------------------------------------------------------------

_policy_store = PolicyStore()

_UNHEALTHY_MODELS: set[str] = set()

_GROQ_MODELS = [
    os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
]
_GEMINI_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    "gemini-3.6-flash",
    "gemini-3.7-flash",
]
# Try models in sequence: Gemini models first, then Groq models — deduplicated
_MODELS_TO_TRY = list(dict.fromkeys(_GEMINI_MODELS + _GROQ_MODELS))


def _get_llm_instance(model_name: str) -> Any:
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if model_name.startswith("gemini"):
        if ChatGoogleGenerativeAI is None:
            raise ImportError(
                "langchain-google-genai is missing. Please run: pip install langchain-google-genai"
            )
        if gemini_key and not gemini_key.startswith("your_"):
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=gemini_key,
                max_retries=0,  # Fail fast on errors to trigger immediate model failover
            )
        raise ValueError("GEMINI_API_KEY is not set or invalid in .env")

    if ChatGroq and groq_key and not groq_key.startswith("your_"):
        return ChatGroq(model=model_name, temperature=0, groq_api_key=groq_key, max_retries=0)

    raise ValueError(
        "No valid LLM API key configured for Groq/Gemini."
    )


def _get_content_text(content: Any) -> str:
    """Normalize LLM response content to a plain string whether it is a str, list of blocks, or dict."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(item["text"])
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content) if content is not None else ""


def invoke_llm(messages: list[Any]) -> Any:
    """Invoke LLM (Google Gemini or Groq) with automatic model fallback."""
    last_exc = None

    for model_name in _MODELS_TO_TRY:
        if model_name in _UNHEALTHY_MODELS:
            continue
        try:
            llm = _get_llm_instance(model_name)
            res = llm.invoke(messages)
            if hasattr(res, "content"):
                res.content = _get_content_text(res.content)
            return res
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            if any(sig in err_str for sig in ("404", "400", "not_found", "notfounderror", "badrequesterror", "invalid_api_key", "unauthorized")):
                logger.warning(
                    "LLM model '%s' returned permanent error (%s: %s); blacklisting for session.",
                    model_name, type(exc).__name__, exc,
                )
                _UNHEALTHY_MODELS.add(model_name)
            else:
                logger.warning(
                    "LLM model '%s' unavailable (%s: %s), trying next fallback model...",
                    model_name, type(exc).__name__, exc,
                )
            continue

    if last_exc:
        raise last_exc


def invoke_llm_stream(messages: list[Any]):
    """Stream token chunks from LLM (Google Gemini or Groq) with automatic model fallback."""
    for model_name in _MODELS_TO_TRY:
        if model_name in _UNHEALTHY_MODELS:
            continue
        try:
            llm = _get_llm_instance(model_name)
            for chunk in llm.stream(messages):
                raw_content = getattr(chunk, "content", None)
                if raw_content:
                    text = _get_content_text(raw_content)
                    if text:
                        yield text
            return
        except Exception as exc:
            err_str = str(exc).lower()
            if any(sig in err_str for sig in ("404", "400", "not_found", "notfounderror", "badrequesterror", "invalid_api_key", "unauthorized")):
                _UNHEALTHY_MODELS.add(model_name)
            logger.warning("LLM stream model '%s' unavailable (%s: %s), trying fallback...", model_name, type(exc).__name__, exc)
            continue


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class BotState(TypedDict):
    # ── Input for this turn ──────────────────────────────────────────────────
    thread_id: str
    user_message: str

    # ── Session-persistent (survive across turns) ─────────────────────────
    last_location: dict[str, Any] | None      # {lat, lon, display_name}
    last_weather: dict[str, Any] | None        # flat Open-Meteo fields
    last_weather_ts: str | None                # ISO timestamp of last fetch
    last_weather_loc: str | None               # Display name of cached location
    last_sop_id: str | None
    last_user_query: str | None                # last activity question across turns

    # ── Within-turn working data ──────────────────────────────────────────
    numeric_candidate_ids: list[str]
    selected_sop_id: str | None
    secondary_sop_id: str | None
    reasoning: str
    reply: str
    weather_facts: dict[str, Any] | None       # only fields used in reply placeholders

    # ── Routing signals ───────────────────────────────────────────────────
    failure_reason: str | None  # "location_not_found" | "no_location_given" | "weather_fetch_failed"


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _route_after_location(state: BotState) -> Literal["fetch_weather", "honest_failure"]:
    if state.get("failure_reason"):
        return "honest_failure"
    return "fetch_weather"


def _route_after_weather(state: BotState) -> Literal["filter_numeric_sops", "honest_failure"]:
    if state.get("failure_reason"):
        return "honest_failure"
    return "filter_numeric_sops"


def _route_after_llm_select(state: BotState) -> Literal["compose_reply", "no_guidance"]:
    if state.get("selected_sop_id"):
        return "compose_reply"
    return "no_guidance"


# ---------------------------------------------------------------------------
# Node: resolve_location
# ---------------------------------------------------------------------------
# Design: mostly deterministic.
#   1. Regex extraction (pure code, no LLM) — handles "in X", "at X", "for X", "near X"
#   2. If regex fails, an LLM call extracts a candidate place name (cannot invent geocode)
#   3. Session reuse if no new place is named
#   4. Geocoding validates everything — LLM output only reaches geocode as a string
# ---------------------------------------------------------------------------

def _failsafe_location_extract(user_message: str) -> str | None:
    """Emergency fallback location extractor used ONLY when ALL external LLM APIs fail (e.g. 429 quota limits or 404)."""
    # 1. Preposition match (rightmost first)
    matches = re.findall(
        r'\b(?:in|at|near|around|from)\s+([A-Za-z\s\-]+?)(?=\s*(?:today|now|this|right|tomorrow|\?|\.|$))',
        user_message,
        re.IGNORECASE,
    )
    for match in reversed(matches):
        candidate = match.strip()
        if candidate and len(candidate.split()) <= 4:
            try:
                geocode(candidate)
                logger.info("_failsafe_location_extract: matched preposition candidate '%s'", candidate)
                return candidate
            except LocationNotFoundError:
                continue

    # 2. Capitalized proper nouns check
    words = [w.strip("?,!.\"':;") for w in user_message.split()]
    stopwords = {
        "is", "it", "safe", "to", "cycle", "drive", "walk", "run", "today", "now", "a",
        "good", "day", "for", "picnic", "the", "in", "at", "from", "near", "what", "how",
        "weather", "can", "i", "you", "we", "my", "your", "are", "there", "any"
    }
    for word in words:
        if word and word[0].isupper() and word.lower() not in stopwords:
            try:
                geocode(word)
                logger.info("_failsafe_location_extract: matched capitalized word '%s'", word)
                return word
            except LocationNotFoundError:
                continue

    return None


def _extract_location_llm(user_message: str) -> str | None:
    """Ask LLM to extract the main city, town, region, or country from the user message."""
    system = (
        "You are a location extraction assistant. "
        "Extract the city, town, region, or country name the user is asking about or traveling from/in. "
        "Ignore generic non-city words like 'mountains', 'beach', 'park', 'work', 'home', 'outside', 'gym', 'hills'. "
        "Reply with ONLY the location name as plain text (e.g. 'Berlin', 'Hyderabad', 'London'). "
        "If no specific city/location name is mentioned in the message, reply with exactly: NONE"
    )
    try:
        response = invoke_llm([
            SystemMessage(content=system),
            HumanMessage(content=user_message),
        ])
        candidate = response.content.strip().strip(" \"'.")
        if not candidate or candidate.upper() == "NONE":
            return None
        return candidate
    except Exception as exc:
        logger.warning("LLM location extraction failed: %s — executing emergency failsafe", exc)
        return _failsafe_location_extract(user_message)

def _resolve_query_context(current_msg: str, previous_query: str | None) -> str:
    """Combine or preserve the activity question when user provides a follow-up location or short answer."""
    if not previous_query or current_msg.strip().lower() == previous_query.strip().lower():
        return current_msg

    activity_keywords = {
        "cycle", "cycling", "bike", "biking", "ride", "riding", "run", "running",
        "jog", "jogging", "exercise", "walk", "picnic", "drive", "driving", "travel",
        "play", "outdoor", "outside", "safe", "safe to", "bouldering", "hiking", "swim"
    }

    current_has_activity = any(w in current_msg.lower() for w in activity_keywords)
    prev_has_activity = any(w in previous_query.lower() for w in activity_keywords)

    if prev_has_activity and not current_has_activity:
        return f"{previous_query} (Location: {current_msg})"

    return current_msg


def resolve_location(state: BotState) -> dict[str, Any]:
    """Resolve location using LLM semantic extraction followed by Open-Meteo geocoding validation.

    LLM is the sole extraction engine — no regex heuristics. This handles any phrasing
    naturally: 'Is it a good day for a picnic in London?' → LLM extracts 'London'.
    """
    user_msg = state["user_message"].strip()
    prev_query = state.get("last_user_query")
    current_query = _resolve_query_context(user_msg, prev_query)
    existing_loc = state.get("last_location")

    # Step 1: Ask LLM to extract the location name from the user message
    candidate = _extract_location_llm(user_msg)

    # Step 2: If LLM found no location, reuse session location if available
    if not candidate:
        if existing_loc:
            logger.info("resolve_location: no new location in message; reusing session location %s", existing_loc.get("display_name"))
            return {
                "last_user_query": current_query,
                "failure_reason": None,
            }
        return {
            "failure_reason": "no_location_given",
            "last_location": None,
            "last_user_query": current_query,
        }

    # Step 4: Geocode candidate extracted by LLM
    try:
        location = geocode(candidate)
        logger.info("resolve_location: geocoded '%s' -> %s OK", candidate, location["display_name"])
        return {
            "last_location": location,
            "last_user_query": current_query,
            "failure_reason": None,
        }
    except LocationNotFoundError:
        return {
            "failure_reason": "location_not_found",
            "last_location": None,
            "last_user_query": current_query,
        }


# ---------------------------------------------------------------------------
# Node: fetch_weather
# ---------------------------------------------------------------------------

def _parse_target_hour(user_message: str) -> int | None:
    """Extract requested target hour (0-23) from user query if a specific time is mentioned."""
    if not user_message:
        return None
    msg = user_message.lower()

    # 1. Match 12-hour format: "11:30 am", "11:30am", "11 am", "3 pm", "3:15pm"
    match12 = re.search(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', msg)
    if match12:
        hour = int(match12.group(1))
        meridiem = match12.group(3)
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        return hour

    # 2. Match 24-hour format: "14:30", "at 14:00"
    match24 = re.search(r'\b([01]?\d|2[0-3]):([0-5]\d)\b', msg)
    if match24:
        return int(match24.group(1))

    # 3. Match general time of day terms
    if any(w in msg for w in ["noon", "midday"]):
        return 12
    if any(w in msg for w in ["afternoon"]):
        return 14
    if any(w in msg for w in ["evening"]):
        return 18
    if any(w in msg for w in ["morning"]):
        return 9
    if any(w in msg for w in ["night"]):
        return 21

    return None


def fetch_weather_node(state: BotState) -> dict[str, Any]:
    """Fetch current or time-specific forecast weather for the resolved location."""
    loc = state["last_location"]
    if not loc:
        return {"failure_reason": "weather_fetch_failed"}

    user_msg = state.get("last_user_query") or state["user_message"]
    target_hour = _parse_target_hour(user_msg)

    # Check session cache (15-minute window ONLY IF LOCATION HAS NOT CHANGED AND NO SPECIFIC TIME WAS REQUESTED)
    last_weather = state.get("last_weather")
    last_weather_ts = state.get("last_weather_ts")
    last_weather_loc = state.get("last_weather_loc")
    current_loc_name = loc.get("display_name")

    if target_hour is None and last_weather and last_weather_ts and last_weather_loc == current_loc_name:
        try:
            ts = datetime.fromisoformat(last_weather_ts)
            age_sec = (datetime.now(timezone.utc) - ts).total_seconds()
            if age_sec < 900:  # 15 minutes
                logger.info("fetch_weather_node: reusing cached weather for '%s' (age=%.1fs)", current_loc_name, age_sec)
                return {"failure_reason": None}
        except Exception:
            pass

    try:
        weather = fetch_weather(loc["lat"], loc["lon"], target_hour=target_hour)
        return {
            "last_weather": weather,
            "last_weather_ts": datetime.now(timezone.utc).isoformat(),
            "last_weather_loc": current_loc_name,
            "failure_reason": None,
        }
    except WeatherFetchError as exc:
        logger.error("fetch_weather_node: %s", exc)
        return {"failure_reason": "weather_fetch_failed"}


# ---------------------------------------------------------------------------
# Node: filter_numeric_sops
# ---------------------------------------------------------------------------

_OP_MAP: dict[str, Any] = {
    ">":  op_module.gt,
    ">=": op_module.ge,
    "<":  op_module.lt,
    "<=": op_module.le,
    "==": op_module.eq,
}


def filter_numeric_sops(state: BotState) -> dict[str, Any]:
    """Evaluate every numeric SOP's condition against fetched weather.
    Returns list of triggered SOP ids. Pure code — no LLM."""
    weather = state.get("last_weather") or {}
    triggered: list[str] = []
    for sop in _policy_store.get_numeric():
        cond = sop.get("condition")
        if not cond:
            continue
        field = cond.get("field")
        op_str = cond.get("operator")
        threshold = cond.get("value")
        if field in weather and weather[field] is not None and op_str in _OP_MAP:
            actual = float(weather[field])
            target = float(threshold)
            if _OP_MAP[op_str](actual, target):
                triggered.append(sop["id"])
    logger.info("filter_numeric_sops: triggered %s", triggered)
    return {"numeric_candidate_ids": triggered}


# ---------------------------------------------------------------------------
# Node: llm_select_sop
# ---------------------------------------------------------------------------

def _build_sop_summary(sop: dict[str, Any]) -> str:
    """Format one SOP for the LLM prompt."""
    return (
        f"  ID: {sop['id']} | severity: {sop['severity']} | "
        f"match_type: {sop['match_type']}\n"
        f"  applies_when: {sop['applies_when']}"
    )


def llm_select_sop(state: BotState) -> dict[str, Any]:
    """LLM call to select the best matching SOP using pure LLM semantic reasoning."""
    weather = state.get("last_weather") or {}
    location = state.get("last_location") or {}
    user_msg = state.get("last_user_query") or state["user_message"]
    numeric_ids = state.get("numeric_candidate_ids") or []

    all_sops = _policy_store.get_all()
    sop_lines = "\n\n".join(_build_sop_summary(s) for s in all_sops)
    numeric_candidates_text = ", ".join(numeric_ids) if numeric_ids else "none"

    system_prompt = (
        "You are a policy-matching assistant. Your only job is to classify which SOP "
        "(from the list provided in this message) applies to the user's question, given "
        "the weather data provided in this message. Output ONLY valid JSON, no markdown:\n"
        '{"sop_id": "SOP-XXX", "matched_condition": "<condition text from that SOP>", "reasoning": "one sentence"}\n'
        "or if truly nothing fits:\n"
        '{"sop_id": null, "reasoning": "one sentence"}\n\n'
        "RULES:\n"
        "1. Only consider SOPs explicitly listed in this message. Never use an SOP id, "
        "condition, or wording that isn't shown to you here.\n"
        "2. Base your reasoning and any figures you cite ONLY on the weather data provided in "
        "this message.\n"
        "3. Numeric-threshold SOPs that code has already verified as triggered are confirmed "
        "candidates. Semantic/situational SOPs (fuzzy match on intent, not a threshold) are equally "
        "valid candidates when the situation described fits.\n"
        "4. If multiple SOPs match, pick the single highest-severity one.\n"
        "5. Return null ONLY if, after checking every SOP listed, none of their conditions are met."
    )

    weather_summary = {
        k: v for k, v in weather.items()
        if v is not None and k in (
            "temperature_2m", "apparent_temperature", "wind_speed_10m",
            "wind_gusts_10m", "precipitation", "uv_index", "surface_pressure",
            "weather_code",
        )
    }

    user_prompt = (
        f"User question: {user_msg}\n"
        f"Location: {location.get('display_name', 'unknown')}\n"
        f"Key weather: {json.dumps(weather_summary)}\n"
        f"Numeric SOPs already triggered (code-verified): {numeric_candidates_text}\n\n"
        f"Available SOPs:\n{sop_lines}\n\n"
        "Output the JSON now:"
    )

    def _parse_sop_response(raw: str) -> tuple[str | None, str]:
        """Parse LLM JSON response. Returns (sop_id, reasoning)."""
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            json_str = raw[start : end + 1]
            try:
                parsed = json.loads(json_str)
                sop_id = parsed.get("sop_id")
                reasoning = parsed.get("reasoning", "")
                if sop_id is not None and _policy_store.get_by_id(sop_id) is None:
                    logger.error("LLM returned unknown sop_id '%s'", sop_id)
                    return None, "LLM returned an invalid SOP id; treating as no match."
                return sop_id, reasoning
            except Exception as e:
                logger.warning("Failed to parse JSON from response: %s", e)
        return None, "No valid JSON object found in response"

    try:
        response = invoke_llm([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        sop_id, reasoning = _parse_sop_response(response.content)

        # ── Secondary SOP (when multiple numerics triggered) ───────────────
        secondary_sop_id: str | None = None
        if sop_id and len(numeric_ids) > 1:
            other_ids = [sid for sid in numeric_ids if sid != sop_id]
            if other_ids:
                runner_up = _policy_store.get_highest_severity(other_ids)
                if runner_up:
                    secondary_sop_id = runner_up["id"]

        logger.info("llm_select_sop → %s (secondary: %s)", sop_id, secondary_sop_id)
        return {
            "selected_sop_id": sop_id,
            "secondary_sop_id": secondary_sop_id,
            "reasoning": reasoning,
        }

    except Exception as exc:
        logger.error("llm_select_sop LLM call failed: %s", exc)
        if numeric_ids:
            fallback = _policy_store.get_highest_severity(numeric_ids)
            if fallback:
                logger.warning("Falling back to highest numeric candidate: %s", fallback["id"])
                return {
                    "selected_sop_id": fallback["id"],
                    "secondary_sop_id": None,
                    "reasoning": "LLM call failed; used highest numeric candidate as fallback",
                }

        # Fallback for semantic SOPs when LLM API is unavailable/rate-limited
        msg_lower = user_msg.lower()
        semantic_fallback_id = None
        if any(w in msg_lower for w in ["cycle", "cycling", "bike", "biking", "run", "running", "jog", "jogging", "exercise", "outdoor"]):
            semantic_fallback_id = "SOP-011"
        elif any(w in msg_lower for w in ["picnic", "sit outside", "eat outside"]):
            semantic_fallback_id = "SOP-010"
        elif any(w in msg_lower for w in ["drive", "driving", "road trip", "travel"]):
            semantic_fallback_id = "SOP-006"

        if semantic_fallback_id:
            fallback = _policy_store.get_by_id(semantic_fallback_id)
            if fallback:
                logger.warning("LLM call failed; using semantic fallback SOP: %s", fallback["id"])
                return {
                    "selected_sop_id": fallback["id"],
                    "secondary_sop_id": None,
                    "reasoning": f"LLM API unavailable; matched semantic fallback SOP {fallback['id']}",
                }

        return {
            "selected_sop_id": None,
            "secondary_sop_id": None,
            "reasoning": f"LLM selection failed: {exc}",
        }


# ---------------------------------------------------------------------------
# Node: compose_reply
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def _substitute_placeholders(template: str, weather: dict[str, Any]) -> tuple[str, dict[str, Any], list[str]]:
    """Replace {{field_name}} placeholders with actual weather values.

    Returns:
        (substituted_text, weather_facts_used, missing_fields)
    weather_facts_used contains only the fields that were actually referenced.
    missing_fields lists any placeholder field names not found in weather.
    """
    weather_facts_used: dict[str, Any] = {}
    missing_fields: list[str] = []

    def replace(match: re.Match) -> str:
        field = match.group(1)
        val = weather.get(field)
        if val is None:
            missing_fields.append(field)
            return f"[{field}: data unavailable]"
        weather_facts_used[field] = val
        # Round floats to 1 decimal for readability
        if isinstance(val, float):
            return str(round(val, 1))
        return str(val)

    result = _PLACEHOLDER_RE.sub(replace, template)
    return result, weather_facts_used, missing_fields


def compose_reply(state: BotState) -> dict[str, Any]:
    """Compose the final reply using the selected SOP's guidance as a template.

    The LLM generates reply text with {{field_name}} placeholders, then
    code substitutes the real fetched values. If a placeholder doesn't
    resolve, we log the bug and fall back to a safe message rather than
    showing a broken placeholder.
    """
    sop_id = state["selected_sop_id"]
    secondary_sop_id = state.get("secondary_sop_id")
    weather = state.get("last_weather") or {}
    location = state.get("last_location") or {}
    user_msg = state.get("last_user_query") or state["user_message"]

    sop = _policy_store.get_by_id(sop_id)
    if sop is None:
        # Should never happen — llm_select_sop validates ids
        logger.error("compose_reply: sop_id '%s' not found in PolicyStore", sop_id)
        return {
            "reply": (
                "I encountered an internal error while preparing your safety advice. "
                "Please try again."
            ),
            "weather_facts": None,
            "last_sop_id": sop_id,
        }

    guidance_template = sop["guidance"]

    # Build the secondary note if applicable
    secondary_note = ""
    if secondary_sop_id:
        secondary_sop = _policy_store.get_by_id(secondary_sop_id)
        if secondary_sop:
            secondary_note = (
                f"\n\nNote: {secondary_sop['applies_when'].split('.')[0].strip()} "
                f"({secondary_sop['id']}, {secondary_sop['severity']} severity) also applies to your situation."
            )

    # Ask LLM to produce the reply using the guidance as a base template
    # The LLM must NOT invent numbers — it uses {{field_name}} placeholders
    system_prompt = f"""You are an outdoor activity weather safety advisor. Write a direct, warm, and helpful answer to the user's question, grounded strictly in the provided SOP Safety Guidance.

STRICT INSTRUCTIONS:
1. Answer the user's specific question directly in natural, friendly conversational prose.
2. Incorporate the core safety advice from the SOP guidance. Do not add outside safety rules.
3. For any weather numbers cited, use ONLY {{{{field_name}}}} placeholder syntax — e.g., {{{{temperature_2m}}}} or {{{{wind_speed_10m}}}}.
4. Available weather placeholders: {', '.join(weather.keys())}
5. Do NOT include markdown code blocks, preamble, or meta-commentary. Output ONLY the response text."""

    user_prompt = f"""User question: {user_msg}
Location: {location.get('display_name', 'your location')}
Matched Policy: {sop['id']} ({sop['category']}, severity: {sop['severity']})

SOP Safety Guidance to follow:
{guidance_template}

Write a clear, helpful response addressing the user's query:"""

    try:
        response = invoke_llm([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        reply_template = response.content.strip()
    except Exception as exc:
        logger.error("compose_reply LLM call failed: %s", exc)
        # Fall back to the raw guidance template from YAML
        reply_template = guidance_template

    # Substitute placeholders with real values
    reply_text, weather_facts, missing = _substitute_placeholders(reply_template, weather)

    if missing:
        logger.error(
            "compose_reply: placeholder fields not found in weather data: %s. "
            "This is a bug — check SOP guidance field names against Open-Meteo API.",
            missing,
        )
        # If substitution is badly broken, fall back to safe message
        if reply_text.count("[") > 3:
            reply_text = (
                "I have safety guidance for your situation but encountered an issue "
                "formatting the weather data. Please try again in a moment."
            )
            weather_facts = {}

    # Append secondary SOP note
    reply_text = reply_text + secondary_note

    return {
        "reply": reply_text,
        "weather_facts": weather_facts,
        "last_sop_id": sop_id,
    }


# ---------------------------------------------------------------------------
# Node: honest_failure  (no LLM)
# ---------------------------------------------------------------------------

_FAILURE_MESSAGES = {
    "location_not_found": (
        "I searched for that location but couldn't find it. "
        "Could you try a different city name or spelling? "
        "For example: \"Is it safe to cycle in Mumbai today?\""
    ),
    "no_location_given": (
        "Which city or area did you have in mind? "
        "I need a location to check the weather. "
        "Try something like: \"Is it safe to cycle in London today?\""
    ),
    "weather_fetch_failed": (
        "I'm unable to retrieve weather data right now — "
        "the weather service may be temporarily unavailable. "
        "Please try again in a moment."
    ),
}
_FAILURE_DEFAULT = (
    "Something went wrong while checking conditions. Please try again."
)


def honest_failure(state: BotState) -> dict[str, Any]:
    """Return a clear, templated failure message. No LLM."""
    reason = state.get("failure_reason", "")
    message = _FAILURE_MESSAGES.get(reason, _FAILURE_DEFAULT)
    return {
        "reply": message,
        "selected_sop_id": None,
        "weather_facts": None,
    }


# ---------------------------------------------------------------------------
# Node: no_guidance  (no LLM)
# ---------------------------------------------------------------------------

def no_guidance(state: BotState) -> dict[str, Any]:
    """Return a clear templated message when no SOP matches. No LLM."""
    # List the categories that DO have guidance (read from PolicyStore — no hardcoding)
    categories = sorted({s["category"] for s in _policy_store.get_all()})
    cat_list = ", ".join(categories)

    message = (
        "I don't have written safety guidance that covers that specific situation. "
        f"My guidance currently covers: {cat_list}. "
        "For questions in those areas, feel free to ask — for example about cycling, "
        "outdoor exercise, travel safety, or weather risks for vulnerable groups."
    )
    return {
        "reply": message,
        "selected_sop_id": None,
        "weather_facts": None,
    }


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Build the LangGraph StateGraph. Call once at startup."""
    builder = StateGraph(BotState)

    # Add nodes
    builder.add_node("resolve_location", resolve_location)
    builder.add_node("fetch_weather", fetch_weather_node)
    builder.add_node("filter_numeric_sops", filter_numeric_sops)
    builder.add_node("llm_select_sop", llm_select_sop)
    builder.add_node("compose_reply", compose_reply)
    builder.add_node("honest_failure", honest_failure)
    builder.add_node("no_guidance", no_guidance)

    # Entry point
    builder.set_entry_point("resolve_location")

    # Edges with conditional routing
    builder.add_conditional_edges(
        "resolve_location",
        _route_after_location,
        {"fetch_weather": "fetch_weather", "honest_failure": "honest_failure"},
    )
    builder.add_conditional_edges(
        "fetch_weather",
        _route_after_weather,
        {"filter_numeric_sops": "filter_numeric_sops", "honest_failure": "honest_failure"},
    )
    builder.add_edge("filter_numeric_sops", "llm_select_sop")
    builder.add_conditional_edges(
        "llm_select_sop",
        _route_after_llm_select,
        {"compose_reply": "compose_reply", "no_guidance": "no_guidance"},
    )
    builder.add_edge("compose_reply", END)
    builder.add_edge("honest_failure", END)
    builder.add_edge("no_guidance", END)

    return builder.compile(checkpointer=checkpointer)


# Module-level compiled graph used by FastAPI
memory = MemorySaver()
graph = build_graph(checkpointer=memory)
