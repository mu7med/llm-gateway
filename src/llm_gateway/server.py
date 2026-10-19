"""OpenAI-compatible FastAPI server exposing the gateway over HTTP.

POST /v1/chat/completions  — same request/response shape as OpenAI, so any
existing OpenAI SDK/agent framework can be pointed at the gateway with a
base_url change. Every call is routed, gated, ledgered.

GET /health — backend health probes.
GET /finops/summary — cost aggregates.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError("install server extras: pip install 'llm-gateway[server]'") from exc

from .config import load_config
from .core import AllBackendsFailedError, CompletionRequest, Gateway
from .finops import CostLedger

app = FastAPI(title="LLM Gateway", version="0.1.0",
              description="Multi-provider LLM gateway with purpose routing, fallback, FinOps.")

_cfg = load_config(os.environ.get("LLM_GATEWAY_CONFIG"))
_ledger = CostLedger(path=os.environ.get("LLM_GATEWAY_LEDGER"))
_gw = Gateway(_cfg, on_attempt=_ledger.record_attempt)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None  # ignored for routing; purpose drives the chain
    purpose: str = Field(default="default", description="Routing purpose")
    messages: list[ChatMessage]
    max_tokens: int = 4096
    temperature: float = 0.7


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    user_msgs = [m.content for m in req.messages if m.role == "user"]
    system_msgs = [m.content for m in req.messages if m.role == "system"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="no user message")
    prompt = user_msgs[-1]
    system = "\n".join(system_msgs) if system_msgs else None

    try:
        result = _gw.complete(CompletionRequest(
            prompt=prompt, purpose=req.purpose, system=system,
            max_tokens=req.max_tokens, temperature=req.temperature))
    except AllBackendsFailedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "id": f"gw-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"{result.backend}:{result.model}",
        "gateway": {
            "purpose": result.purpose,
            "fallback_used": result.fallback_used,
            "attempts_failed": result.attempts_failed,
            "cost_usd": result.cost_usd,
            "elapsed_seconds": result.elapsed_seconds,
        },
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": result.content}}],
        "usage": {"prompt_tokens": result.tokens_in,
                  "completion_tokens": result.tokens_out,
                  "total_tokens": result.tokens_total},
    }


@app.get("/health")
def health():
    return _gw.health_check()


@app.get("/finops/summary")
def finops_summary():
    return {"by_model": _ledger.summary(), "by_purpose": _ledger.summary_by_purpose()}
