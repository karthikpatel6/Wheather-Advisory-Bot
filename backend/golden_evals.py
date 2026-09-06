"""
golden_evals.py — Automated Golden Dataset Evaluator for Weather Advisory Bot.

Runs 20 curated test scenarios from golden_dataset.json covering:
  - High-Wind Hazards (cycling, road travel)
  - Extreme Heat & Sun Exposure (running, vulnerable groups, children)
  - Heavy Rain & Severe Storm Systems
  - Paraphrased Intent (reworded questions without exact SOP keywords)
  - Multi-Hazard Conflict Resolution (severity prioritization)
  - Uncovered Activities (indoor bouldering, indoor gaming -> honest no_guidance)
  - Adversarial Prompt Injections
  - System Outages & Geocoding Failures
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import unittest.mock as mock
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).parent))

from graph import BotState, build_graph
from policy_store import PolicyStore
from weather import LocationNotFoundError, WeatherFetchError

_store = PolicyStore()


def _make_weather(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "temperature_2m": 20.0,
        "apparent_temperature": 20.0,
        "relative_humidity_2m": 50.0,
        "wind_speed_10m": 12.0,
        "wind_gusts_10m": 18.0,
        "precipitation": 0.0,
        "rain": 0.0,
        "showers": 0.0,
        "snowfall": 0.0,
        "weather_code": 1,
        "cloud_cover": 15.0,
        "uv_index": 3.0,
        "visibility": 20000.0,
        "surface_pressure": 1013.0,
    }
    if overrides:
        base.update(overrides)
    return base


def _run_graph(message: str, extra_state: dict | None = None) -> dict[str, Any]:
    import asyncio
    g = build_graph(checkpointer=None)
    state: dict[str, Any] = {
        "thread_id": "golden-eval",
        "user_message": message,
        "last_location": None,
        "last_weather": None,
        "last_weather_ts": None,
        "last_sop_id": None,
        "numeric_candidate_ids": [],
        "selected_sop_id": None,
        "secondary_sop_id": None,
        "reasoning": "",
        "reply": "",
        "weather_facts": None,
        "failure_reason": None,
    }
    if extra_state:
        state.update(extra_state)

    config = {"configurable": {"thread_id": "golden-eval"}}

    async def _invoke():
        return await g.ainvoke(state, config=config)

    return asyncio.run(_invoke())


def _numbers_in_text(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def run_golden_evals():
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print("=" * 80)
    print(f"  WEATHER ADVISORY BOT — GOLDEN DATASET BENCHMARK ({len(cases)} SCENARIOS)")
    print("=" * 80)

    results: list[dict[str, Any]] = []

    for case in cases:
        case_id = case["id"]
        cat = case["category"]
        query = case["query"]
        expected_sop = case["expected_sop_id"]
        sim_error = case.get("simulated_error")

        loc_name = case.get("location", "Test City")
        mock_loc = {"lat": 51.5074, "lon": -0.1278, "display_name": loc_name}
        mock_w = _make_weather(case.get("mock_weather"))

        actual_sop = None
        reply = ""
        passed = False
        notes = []

        try:
            if sim_error == "WeatherFetchError":
                with (
                    mock.patch("graph.geocode", return_value=mock_loc),
                    mock.patch("graph.fetch_weather", side_effect=WeatherFetchError("Outage")),
                ):
                    res = _run_graph(query)
            elif sim_error == "LocationNotFoundError":
                with (
                    mock.patch("graph.geocode", side_effect=LocationNotFoundError("Not found")),
                    mock.patch("graph._extract_location_llm", return_value="NonExistentCity123"),
                ):
                    res = _run_graph(query)
            else:
                with (
                    mock.patch("graph.geocode", return_value=mock_loc),
                    mock.patch("graph.fetch_weather", return_value=mock_w),
                ):
                    res = _run_graph(query)

            actual_sop = res.get("selected_sop_id")
            reply = res.get("reply", "")
            weather_used = res.get("weather_facts") or {}

            # Assertion Logic based on scenario category
            if sim_error:
                passed = actual_sop is None and any(kw in reply.lower() for kw in ["unable", "unavailable", "find", "couldn't", "search", "retry"])
                notes.append(f"Simulated error '{sim_error}'. Returned honest fallback message.")
            elif expected_sop is None:
                # Uncovered activity or injection
                no_injection = "sop-099" not in reply.lower()
                no_invented = not any(p in reply.lower() for p in ["sop-", "according to our policy"]) if "bouldering" in query or "games" in query else True
                passed = (actual_sop is None or actual_sop != "SOP-099") and no_injection and no_invented
                notes.append(f"Expected no SOP. Received sop_id={actual_sop}.")
            else:
                sop_match = actual_sop == expected_sop
                passed = sop_match
                notes.append(f"Expected {expected_sop}, got {actual_sop}.")

        except Exception as exc:
            passed = False
            notes.append(f"Exception during run: {exc}")

        results.append({
            "id": case_id,
            "category": cat,
            "query": query,
            "expected_sop": expected_sop,
            "actual_sop": actual_sop,
            "passed": passed,
            "reply_snippet": reply[:100].replace("\n", " ") + "..." if len(reply) > 100 else reply,
            "notes": " ".join(notes),
        })

    # Print Detailed Report
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)

    print()
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"[{r['id']:02d}] {status} | Cat: {r['category']:<22} | Exp: {str(r['expected_sop']):<8} | Act: {str(r['actual_sop']):<8}")
        print(f"     Query: {r['query']!r}")
        print(f"     Reply: {r['reply_snippet']!r}")
        print("─" * 80)

    print()
    print("=" * 80)
    print(f"  BENCHMARK SUMMARY: {passed_count}/{total_count} PASSED ({passed_count/total_count*100:.1f}%)")
    print("=" * 80)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_golden_evals()
    sys.exit(0 if success else 1)
