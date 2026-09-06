"""
main.py — FastAPI application for the Weather Advisory Bot.

Single endpoint:
    POST /chat
    Request:  {"thread_id": str, "message": str}
    Response: {"reply": str, "sop_id": str | null, "weather_used": dict | null}

The graph is compiled once at startup with MemorySaver for session persistence.
Each thread_id gets its own checkpointed state.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Import the compiled graph (loads PolicyStore + Groq client at import time)
from graph import graph  # noqa: E402

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Weather Advisory Bot",
    description=(
        "Answers outdoor-activity safety questions using live Open-Meteo weather data "
        "matched against a fixed set of policy rules (SOPs)."
    ),
    version="1.0.0",
)

# Allow requests from the Vite dev server and any localhost port
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session identifier — all turns with the same id share state",
    )
    message: str = Field(..., min_length=1, description="User's message")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Bot's reply to the user")
    sop_id: str | None = Field(
        None,
        description="ID of the matched SOP (e.g. SOP-001), or null if none matched",
    )
    weather_used: dict[str, Any] | None = Field(
        None,
        description=(
            "The weather fields that were substituted into the reply placeholders. "
            "Exactly the fields cited in the reply — useful for diffing reply numbers "
            "against the actual API response."
        ),
    )


import asyncio
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Endpoint: POST /chat (synchronous)
# ---------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a user message and return a safety advisory reply."""
    logger.info("POST /chat thread_id=%s message=%r", request.thread_id, request.message[:80])

    initial_state = {
        "thread_id": request.thread_id,
        "user_message": request.message,
        "last_location": None,
        "last_weather": None,
        "last_weather_ts": None,
        "last_sop_id": None,
        "last_user_query": None,
        "numeric_candidate_ids": [],
        "selected_sop_id": None,
        "secondary_sop_id": None,
        "reasoning": "",
        "reply": "",
        "weather_facts": None,
        "failure_reason": None,
    }

    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("Unhandled error in graph.ainvoke: %s", exc)
        return ChatResponse(
            reply=(
                "An unexpected error occurred. Please try again. "
                "If this persists, check the server logs."
            ),
            sop_id=None,
            weather_used=None,
        )

    reply = final_state.get("reply", "")
    sop_id = final_state.get("selected_sop_id")
    weather_facts = final_state.get("weather_facts")

    logger.info(
        "Response: sop_id=%s weather_fields=%s reply_len=%d",
        sop_id,
        list(weather_facts.keys()) if weather_facts else None,
        len(reply),
    )

    return ChatResponse(
        reply=reply,
        sop_id=sop_id,
        weather_used=weather_facts,
    )


# ---------------------------------------------------------------------------
# Endpoint: POST /chat/stream (SSE Streaming)
# ---------------------------------------------------------------------------


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream safety advisory reply token-by-token using Server-Sent Events (SSE)."""
    logger.info("POST /chat/stream thread_id=%s message=%r", request.thread_id, request.message[:80])

    async def event_generator():
        initial_state = {
            "thread_id": request.thread_id,
            "user_message": request.message,
            "last_location": None,
            "last_weather": None,
            "last_weather_ts": None,
            "last_sop_id": None,
            "last_user_query": None,
            "numeric_candidate_ids": [],
            "selected_sop_id": None,
            "secondary_sop_id": None,
            "reasoning": "",
            "reply": "",
            "weather_facts": None,
            "failure_reason": None,
        }
        config = {"configurable": {"thread_id": request.thread_id}}

        try:
            final_state = await graph.ainvoke(initial_state, config=config)
        except Exception as exc:
            logger.exception("Unhandled error in streaming graph.ainvoke: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'content': 'An unexpected error occurred.'})}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
            return

        reply = final_state.get("reply", "")
        sop_id = final_state.get("selected_sop_id")
        weather_facts = final_state.get("weather_facts")

        # Event 1: Send metadata (sop_id & weather_used) immediately
        meta_payload = {
            "type": "meta",
            "sop_id": sop_id,
            "weather_used": weather_facts,
        }
        yield f"data: {json.dumps(meta_payload)}\n\n"

        # Event 2+: Stream words token-by-token for smooth, typewriter effect
        words = reply.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else word + " "
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            await asyncio.sleep(0.015)  # 15ms smooth cadence

        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# SOPs metadata endpoint (read-only; used by frontend for badge severity lookup)
# ---------------------------------------------------------------------------

from graph import _policy_store  # noqa: E402


@app.get("/sops")
async def list_sops() -> list[dict]:
    """Return all SOP metadata (id, category, severity, match_type, applies_when).
    Guidance text is excluded to keep the payload small.
    """
    return [
        {
            "id": s["id"],
            "category": s["category"],
            "severity": s["severity"],
            "match_type": s["match_type"],
            "applies_when": s["applies_when"][:120] + "…"  # truncated for UI
            if len(s["applies_when"]) > 120
            else s["applies_when"],
        }
        for s in _policy_store.get_all()
    ]
