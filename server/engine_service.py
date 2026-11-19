"""
Thread-owned serving wrapper around :class:`CustomInferenceEngine`.

The engine (scheduler, block manager, CUDA graphs) is *not* thread-safe, so a
single background thread owns it end-to-end:

    - builds the model + engine inside that thread (CUDA graphs are captured
      and replayed on the same thread),
    - runs the continuous-batching ``step()`` loop whenever work is pending,
    - after every step, pushes newly generated token deltas to each request's
      subscriber (an ``asyncio.Queue`` living on the web server's event loop).

HTTP handlers only call :meth:`EngineService.submit` / :meth:`EngineService.abort`,
which take a lock that is never held across a whole generation, only across a
single engine iteration.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """One event delivered to a request subscriber."""
    kind: str                      # "token" | "done" | "error"
    text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Tracked:
    seq_id: str
    seq: Any                        # core.scheduler.Sequence
    loop: asyncio.AbstractEventLoop
    queue: "asyncio.Queue[StreamEvent]"
    prompt_len: int
    sent_tokens: int = 0            # tokens already accounted for
    sent_text: str = ""             # text already delivered


# ---------------------------------------------------------------------------
# Default builder: real Qwen2 model + tokenizer from a HF directory
# ---------------------------------------------------------------------------

def build_engine_from_settings():
    """Build ``(engine, tokenizer, info)`` from ``config.SETTINGS`` / env vars.

    Mirrors the model-loading cell of ``notebooks/benchmark_colab.ipynb``.
    """
    import torch
    from transformers import AutoTokenizer

    from config import reload_settings
    from core.engine import CustomInferenceEngine
    from models.qwen2_model import Qwen2Config, Qwen2Model
    from models.weight_loader import load_qwen2_weights

    settings = reload_settings(override=False)
    model_dir = settings.model_dir
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    device = settings.resolve_device()
    dtype = settings.resolve_torch_dtype()
    config = Qwen2Config.from_hf_config(raw_config, dtype=dtype, max_model_len=settings.max_model_len)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = Qwen2Model(config).to(device=device, dtype=dtype)
    finally:
        torch.set_default_dtype(prev_dtype)
    load_qwen2_weights(model, model_dir, dtype=dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for tok_str in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)

    engine = CustomInferenceEngine(
        model=model,
        num_blocks=settings.num_blocks,
        block_size=settings.block_size,
        max_batch_size=settings.max_batch_size,
        max_batched_tokens=settings.max_batched_tokens,
        max_model_len=settings.max_model_len,
        stop_token_ids=stop_ids,
        enable_prefix_caching=settings.enable_prefix_caching,
        use_cuda_graph=settings.use_cuda_graph and device.startswith("cuda"),
        temperature=settings.temperature,
    )

    gpu = "cpu"
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    info = {
        "model": settings.model_name,
        "model_repo": settings.model_repo,
        "device": device,
        "gpu": gpu,
        "dtype": str(dtype).replace("torch.", ""),
        "max_model_len": engine.max_model_len,
        "max_batch_size": settings.max_batch_size,
        "temperature": settings.temperature,
        "params_b": round(model.num_parameters / 1e9, 3),
    }
    return engine, tokenizer, info


# ---------------------------------------------------------------------------
# Engine service
# ---------------------------------------------------------------------------

class EngineService:
    """Owns the engine on a background thread and streams tokens to subscribers."""

    def __init__(self, builder: Callable[[], tuple] = build_engine_from_settings, idle_wait_s: float = 0.05):
        self._builder = builder
        self._idle_wait_s = idle_wait_s
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tracked: Dict[str, _Tracked] = {}

        self.engine = None
        self.tokenizer = None
        self.info: Dict[str, Any] = {}
        self.status = "starting"          # starting | loading | ready | error
        self.error: Optional[str] = None
        self.ready = threading.Event()
        self.started_at = time.time()
        self.total_requests = 0
        self.total_generated_tokens = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="engine-loop", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wakeup.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        self.status = "loading"
        try:
            t0 = time.perf_counter()
            self.engine, self.tokenizer, self.info = self._builder()
            self.info["load_time_s"] = round(time.perf_counter() - t0, 2)
            self.status = "ready"
            self.ready.set()
            print(f"[server] Engine ready in {self.info['load_time_s']}s: {self.info}")
        except Exception as exc:  # pragma: no cover - depends on runtime
            self.status = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self.ready.set()
            return

        while not self._stop.is_set():
            with self._lock:
                busy = self.engine.has_unfinished_requests()
                if busy:
                    self._step_locked()
            if not busy:
                self._wakeup.wait(self._idle_wait_s)
                self._wakeup.clear()

    # ------------------------------------------------------------------ public API
    def submit(self, prompt_tokens: List[int], max_tokens: int,
               loop: asyncio.AbstractEventLoop) -> tuple[str, "asyncio.Queue[StreamEvent]"]:
        """Queue a request; returns ``(seq_id, queue)``. Raises ValueError on bad input."""
        if self.status != "ready":
            raise RuntimeError(f"Engine not ready (status={self.status}{': ' + self.error if self.error else ''})")
        queue: "asyncio.Queue[StreamEvent]" = asyncio.Queue()
        with self._lock:
            seq_id = self.engine.add_request(prompt_tokens, max_tokens=max_tokens)
            seq = self._find_seq(seq_id)
            self._tracked[seq_id] = _Tracked(
                seq_id=seq_id, seq=seq, loop=loop, queue=queue, prompt_len=len(prompt_tokens),
            )
            self.total_requests += 1
        self._wakeup.set()
        return seq_id, queue

    def abort(self, seq_id: str) -> None:
        """Cancel a request (e.g. client disconnected) and free its KV blocks."""
        with self._lock:
            tr = self._tracked.pop(seq_id, None)
            if tr is None:
                return
            sched = self.engine.scheduler
            if tr.seq in sched.waiting_queue:
                sched.waiting_queue.remove(tr.seq)
                sched.block_manager.free_sequence(seq_id)
            elif tr.seq in sched.running_queue:
                sched.finish_sequence(tr.seq)

    def stats(self) -> Dict[str, Any]:
        base = {
            "status": self.status,
            "error": self.error,
            "uptime_s": round(time.time() - self.started_at, 1),
            "total_requests": self.total_requests,
            "total_generated_tokens": self.total_generated_tokens,
            **self.info,
        }
        if self.status == "ready":
            with self._lock:
                s = self.engine.get_stats()
                base.update({
                    "active_requests": len(self._tracked),
                    "waiting": len(self.engine.scheduler.waiting_queue),
                    "running": len(self.engine.scheduler.running_queue),
                    "kv_block_usage": round(s["kv_block_usage"], 4),
                    "prefix_cache_hit_rate": round(s["prefix_cache_hit_rate"], 4),
                    "cuda_graph_replays": s["cuda_graph_replays"],
                    "eager_decode_steps": s["eager_decode_steps"],
                    "kv_cache_mb": round(s["kv_cache_mb"], 1),
                    "peak_allocated_mb": round(s["peak_allocated_mb"], 1),
                })
        return base

    # ------------------------------------------------------------------ internals
    def _find_seq(self, seq_id: str):
        for seq in reversed(self.engine.scheduler.waiting_queue):
            if seq.seq_id == seq_id:
                return seq
        raise RuntimeError(f"Sequence {seq_id} not found after add_request")

    @staticmethod
    def _emit(tr: _Tracked, event: StreamEvent) -> None:
        try:
            tr.loop.call_soon_threadsafe(tr.queue.put_nowait, event)
        except RuntimeError:
            pass  # event loop closed (server shutting down)

    def _decode_delta(self, tr: _Tracked, tokens: List[int], final: bool) -> str:
        """Incremental detokenization that never splits a multi-byte character."""
        stop_ids = self.engine.stop_token_ids
        visible = tokens[:-1] if tokens and tokens[-1] in stop_ids else tokens
        text = self.tokenizer.decode(visible, skip_special_tokens=True)
        if not final and text.endswith("\ufffd"):
            return ""  # wait for the rest of the UTF-8 sequence
        # If the tokenizer re-merged earlier text, fall back to the length-based tail.
        delta = text[len(tr.sent_text):]
        tr.sent_text = text
        return delta

    def _step_locked(self) -> None:
        try:
            finished = self.engine.step()
        except Exception as exc:  # pragma: no cover - depends on runtime
            traceback.print_exc()
            self._fail_all(f"{type(exc).__name__}: {exc}")
            return

        finished_by_id = {o.seq_id: o for o in finished}

        for seq_id, tr in list(self._tracked.items()):
            out = finished_by_id.get(seq_id)
            tokens = out.output_tokens if out is not None else tr.seq.output_tokens
            if len(tokens) > tr.sent_tokens or out is not None:
                delta = self._decode_delta(tr, tokens, final=out is not None)
                self.total_generated_tokens += len(tokens) - tr.sent_tokens
                tr.sent_tokens = len(tokens)
                if delta:
                    self._emit(tr, StreamEvent("token", text=delta))
            if out is not None:
                stopped = bool(out.output_tokens) and out.output_tokens[-1] in self.engine.stop_token_ids
                self._emit(tr, StreamEvent("done", data={
                    "finish_reason": "stop" if stopped else "length",
                    "prompt_tokens": out.prompt_len,
                    "completion_tokens": out.output_len,
                    "cached_tokens": out.cached_tokens,
                    "ttft_ms": round(out.ttft_ms, 2),
                    "tpot_ms": round(out.tpot_ms, 2),
                    "e2e_ms": round(out.e2e_ms, 2),
                    "tokens_per_s": round(out.output_len / max(out.e2e_ms / 1000, 1e-9), 2),
                    "num_preemptions": out.num_preemptions,
                }))
                del self._tracked[seq_id]

    def _fail_all(self, message: str) -> None:
        """Abort every in-flight request after an engine error so the loop keeps serving."""
        sched = self.engine.scheduler
        for seq in list(sched.running_queue):
            sched.finish_sequence(seq)
        for seq in list(sched.waiting_queue):
            sched.block_manager.free_sequence(seq.seq_id)
        sched.waiting_queue.clear()
        for tr in self._tracked.values():
            self._emit(tr, StreamEvent("error", text=message))
        self._tracked.clear()
