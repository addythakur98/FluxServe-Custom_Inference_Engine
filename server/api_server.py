"""
HTTP API for the Custom Inference Engine (run on the GPU box, e.g. Colab A100).

Endpoints (OpenAI-compatible where it matters, so any OpenAI client works too):

    GET  /health                  liveness + engine status (no auth)
    GET  /v1/models               model list
    GET  /v1/stats                engine / KV-cache / CUDA-graph telemetry
    POST /v1/chat/completions     chat (``"stream": true`` -> Server-Sent Events)
    POST /v1/completions          raw prompt completion (same streaming support)

Auth: if ``ENGINE_API_KEY`` is set, every ``/v1/*`` call needs
``Authorization: Bearer <key>``. Always set it when exposing via a public tunnel.

Run::

    ENGINE_MODEL_DIR=/path/to/qwen2.5-coder-0.5b-instruct \
    ENGINE_API_KEY=secret uvicorn server.api_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List, Literal, Optional

# Make the repo root importable (core/, models/, kernels/, config.py)
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from server.engine_service import EngineService, StreamEvent

API_KEY = os.environ.get("ENGINE_API_KEY", "").strip()
CORS_ORIGINS = [o.strip() for o in os.environ.get("SERVER_CORS_ORIGINS", "*").split(",") if o.strip()]
DEFAULT_MAX_TOKENS = int(os.environ.get("SERVER_DEFAULT_MAX_TOKENS", "512"))
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "SERVER_SYSTEM_PROMPT",
    "You are Qwen, created by Alibaba Cloud. You are a helpful coding assistant.",
)

service: Optional[EngineService] = None


def get_service() -> EngineService:
    """Return the global service, creating (but not starting) it on first use."""
    global service
    if service is None:
        service = EngineService()
    return service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    svc = get_service()
    svc.start()
    yield
    svc.stop()


app = FastAPI(title="Custom Inference Engine API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(min_length=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = False
    model: Optional[str] = None  # accepted for OpenAI compatibility; ignored


class CompletionRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = False
    model: Optional[str] = None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

async def require_api_key(request: Request) -> None:
    if not API_KEY:
        return
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _ready_service() -> EngineService:
    svc = get_service()
    if svc.status != "ready":
        detail = f"Engine is {svc.status}"
        if svc.error:
            detail += f": {svc.error}"
        raise HTTPException(status_code=503, detail=detail)
    return svc


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_chat_prompt(svc: EngineService, messages: List[ChatMessage], max_tokens: int) -> tuple[List[int], int]:
    """Apply the chat template, dropping the oldest turns until the prompt fits."""
    tok = svc.tokenizer
    max_len = svc.engine.max_model_len
    msgs = [m.model_dump() for m in messages]
    if msgs[0]["role"] != "system":
        msgs.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    # Reserve room for at least a short answer.
    max_tokens = max(1, min(max_tokens, max_len - 16))
    while True:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        ids = list(ids)
        if len(ids) + max_tokens < max_len:
            return ids, max_tokens
        # Drop the oldest non-system turn; keep at least the latest user message.
        if len(msgs) > 2:
            msgs.pop(1)
            continue
        # A single huge message: shrink the answer budget, else reject.
        if len(ids) < max_len - 16:
            return ids, max_len - len(ids) - 1
        raise HTTPException(
            status_code=400,
            detail=f"Prompt is {len(ids)} tokens; max_model_len is {max_len}.",
        )


def _build_completion_prompt(svc: EngineService, prompt: str, max_tokens: int) -> tuple[List[int], int]:
    ids = svc.tokenizer.encode(prompt)
    max_len = svc.engine.max_model_len
    if len(ids) >= max_len - 1:
        raise HTTPException(status_code=400, detail=f"Prompt is {len(ids)} tokens; max_model_len is {max_len}.")
    return ids, max(1, min(max_tokens, max_len - len(ids) - 1))


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _submit(svc: EngineService, ids: List[int], max_tokens: int):
    try:
        return svc.submit(ids, max_tokens, asyncio.get_running_loop())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _usage(done: dict) -> dict:
    return {
        "prompt_tokens": done.get("prompt_tokens", 0),
        "completion_tokens": done.get("completion_tokens", 0),
        "total_tokens": done.get("prompt_tokens", 0) + done.get("completion_tokens", 0),
    }


async def _collect(svc: EngineService, seq_id: str, queue: "asyncio.Queue[StreamEvent]") -> tuple[str, dict]:
    parts: List[str] = []
    finished = False
    try:
        while True:
            ev = await queue.get()
            if ev.kind == "token":
                parts.append(ev.text)
            elif ev.kind == "done":
                finished = True
                return "".join(parts), ev.data
            else:
                finished = True
                raise HTTPException(status_code=500, detail=ev.text)
    finally:
        if not finished:
            svc.abort(seq_id)


async def _sse(svc: EngineService, seq_id: str, queue: "asyncio.Queue[StreamEvent]",
               chunk_fn, model_name: str) -> AsyncIterator[str]:
    """Translate engine events into OpenAI-style SSE chunks."""
    finished = False
    created = int(time.time())
    req_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    try:
        yield "data: " + json.dumps(chunk_fn(req_id, created, model_name, "", None, first=True)) + "\n\n"
        while True:
            ev = await queue.get()
            if ev.kind == "token":
                yield "data: " + json.dumps(chunk_fn(req_id, created, model_name, ev.text, None)) + "\n\n"
            elif ev.kind == "done":
                finished = True
                final = chunk_fn(req_id, created, model_name, "", ev.data["finish_reason"])
                final["usage"] = _usage(ev.data)
                final["metrics"] = ev.data
                yield "data: " + json.dumps(final) + "\n\n"
                yield "data: [DONE]\n\n"
                return
            else:
                finished = True
                yield "data: " + json.dumps({"error": {"message": ev.text}}) + "\n\n"
                return
    finally:
        if not finished:  # client disconnected / cancelled mid-stream
            svc.abort(seq_id)


def _chat_chunk(req_id, created, model, text, finish_reason, first=False):
    delta = {"role": "assistant", "content": ""} if first else ({"content": text} if text else {})
    return {
        "id": req_id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _completion_chunk(req_id, created, model, text, finish_reason, first=False):
    return {
        "id": req_id, "object": "text_completion", "created": created, "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
    }


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    svc = get_service()
    return {"name": "custom-inference-engine", "status": svc.status, "model": svc.info.get("model"),
            "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health():
    svc = get_service()
    code = 200 if svc.status == "ready" else 503
    return JSONResponse(status_code=code, content={
        "status": svc.status, "error": svc.error,
        "model": svc.info.get("model"), "gpu": svc.info.get("gpu"),
        "auth_required": bool(API_KEY),
    })


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models():
    svc = get_service()
    name = svc.info.get("model", "loading")
    return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "custom-inference-engine"}]}


@app.get("/v1/stats", dependencies=[Depends(require_api_key)])
async def stats():
    return get_service().stats()


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(req: ChatRequest):
    svc = _ready_service()
    ids, max_tokens = _build_chat_prompt(svc, req.messages, req.max_tokens or DEFAULT_MAX_TOKENS)
    seq_id, queue = _submit(svc, ids, max_tokens)
    model = svc.info.get("model", "custom")

    if req.stream:
        return StreamingResponse(_sse(svc, seq_id, queue, _chat_chunk, model),
                                 media_type="text/event-stream", headers=_SSE_HEADERS)

    text, done = await _collect(svc, seq_id, queue)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": done["finish_reason"]}],
        "usage": _usage(done), "metrics": done,
    }


@app.post("/v1/completions", dependencies=[Depends(require_api_key)])
async def completions(req: CompletionRequest):
    svc = _ready_service()
    ids, max_tokens = _build_completion_prompt(svc, req.prompt, req.max_tokens or DEFAULT_MAX_TOKENS)
    seq_id, queue = _submit(svc, ids, max_tokens)
    model = svc.info.get("model", "custom")

    if req.stream:
        return StreamingResponse(_sse(svc, seq_id, queue, _completion_chunk, model),
                                 media_type="text/event-stream", headers=_SSE_HEADERS)

    text, done = await _collect(svc, seq_id, queue)
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}", "object": "text_completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": done["finish_reason"]}],
        "usage": _usage(done), "metrics": done,
    }

