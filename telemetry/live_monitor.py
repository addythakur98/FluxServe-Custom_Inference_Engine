"""
Live Traffic Replay & Terminal/Notebook Telemetry Monitor.

Streams real conversational requests (loaded from the ShareGPT dataset on
HuggingFace Hub, with an automatic built-in multi-domain fallback trace)
into ``CustomInferenceEngine`` using a Poisson arrival process, while
rendering a live, self-refreshing dashboard that shows:

    - Incoming request arrivals, source dataset/domain, user ID, & QPS
    - Waiting queue vs. active Continuous Batching slots (Prefill vs. Decode)
    - Paged KV Cache block occupancy bar + Prefix Cache Radix Tree hit rate
    - Live token-by-token streaming output for every in-flight sequence
    - Recently completed requests with TTFT, TPOT, and cached-token counts

Works in both:
    1. **Google Colab / Jupyter** (renders a rich dark-mode terminal UI card
       in-place via ``IPython.display``)
    2. **Standard terminal** (renders with ``rich.live.Live`` or ANSI clear)
"""

from __future__ import annotations

import html
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# ShareGPT / Multi-Source Request Trace
# ---------------------------------------------------------------------------

SHAREGPT_HF_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/"
    "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
)

# Shared system prompts (> 64 tokens = 4+ full blocks of 16 tokens) so that
# ShareGPT requests in the same category visibly trigger multi-block Prefix Cache hits!
CATEGORY_SYSTEM_PROMPTS: Dict[str, str] = {
    "ShareGPT/Coding": (
        "You are an expert principal software engineer and systems programmer on ShareGPT. "
        "Provide concise, idiomatic, production-ready Python, C++, and distributed systems code "
        "with clear complexity analysis, edge-case handling, memory-safety notes, and brief "
        "architectural explanations tailored for senior engineers.\n\n"
    ),
    "ShareGPT/Science": (
        "You are a knowledgeable physics, chemistry, and biology professor on ShareGPT. "
        "Explain complex natural phenomena, relativistic mechanics, quantum electrodynamics, "
        "and molecular biology clearly and accurately in plain English while preserving "
        "rigorous first-principles physical intuition.\n\n"
    ),
    "ShareGPT/SysDesign": (
        "You are a principal distributed-systems architect on ShareGPT specializing in "
        "high-throughput, low-tail-latency cloud infrastructure. Analyze scalability, "
        "p99 latency, caching hierarchies, consensus protocols, replication, backpressure, "
        "and fault-tolerance trade-offs with concrete production numbers.\n\n"
    ),
    "ShareGPT/Math&ML": (
        "You are a staff machine-learning researcher and GPU kernel engineer on ShareGPT. "
        "Explain mathematical intuition, optimization dynamics, numerical stability, "
        "FlashAttention SRAM tiling, PagedAttention virtual memory, and deep-learning "
        "transformer architectures succinctly and rigorously.\n\n"
    ),
}

BUILTIN_SHAREGPT_CONVERSATIONS: List[Dict[str, str]] = [
    {"id": "sgpt_901a", "source": "ShareGPT/Coding",    "user": "u_4821", "prompt": "Write a Python decorator that retries a function with exponential backoff."},
    {"id": "sgpt_902b", "source": "ShareGPT/Coding",    "user": "u_1094", "prompt": "How do I implement an LRU cache in Python without functools?"},
    {"id": "sgpt_903c", "source": "ShareGPT/Science",   "user": "u_7732", "prompt": "Why does time slow down near a black hole according to general relativity?"},
    {"id": "sgpt_904d", "source": "ShareGPT/SysDesign", "user": "u_3301", "prompt": "Our p99 API latency spiked from 40ms to 600ms under load. How should we triage it?"},
    {"id": "sgpt_905e", "source": "ShareGPT/Math&ML",   "user": "u_6150", "prompt": "Explain why PagedAttention eliminates external fragmentation in LLM KV caches."},
    {"id": "sgpt_906f", "source": "ShareGPT/Coding",    "user": "u_8842", "prompt": "Explain Python generators vs list comprehensions for streaming a 50GB log file."},
    {"id": "sgpt_907g", "source": "ShareGPT/Math&ML",   "user": "u_2298", "prompt": "What is the online softmax trick in FlashAttention-2 and why does it save HBM bandwidth?"},
    {"id": "sgpt_908h", "source": "ShareGPT/SysDesign", "user": "u_5519", "prompt": "Compare consistent hashing with rendezvous hashing for a distributed KV store."},
    {"id": "sgpt_909i", "source": "ShareGPT/Science",   "user": "u_9910", "prompt": "How do mRNA vaccines train the adaptive immune system without altering DNA?"},
    {"id": "sgpt_910j", "source": "ShareGPT/Coding",    "user": "u_3107", "prompt": "Write a lock-free ring buffer sketch or explain how asyncio queues work in Python."},
    {"id": "sgpt_911k", "source": "ShareGPT/Math&ML",   "user": "u_4402", "prompt": "How does Grouped-Query Attention (GQA) reduce KV cache memory compared to MHA?"},
    {"id": "sgpt_912l", "source": "ShareGPT/SysDesign", "user": "u_7185", "prompt": "When should a microservice use gRPC streaming instead of REST over HTTP/2?"},
    {"id": "sgpt_913m", "source": "ShareGPT/Science",   "user": "u_1643", "prompt": "What happens inside a star right before a core-collapse supernova?"},
    {"id": "sgpt_914n", "source": "ShareGPT/Math&ML",   "user": "u_8024", "prompt": "Why do CUDA Graphs speed up single-token LLM decode so dramatically on T4 GPUs?"},
]


@dataclass
class TrafficRequest:
    """One request in a replay trace."""
    trace_id: str
    source: str
    user: str
    user_prompt: str
    full_prompt: str
    arrival_offset_s: float = 0.0
    max_new_tokens: int = 36


def _classify_prompt(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("python", "code", "function", "bug", "sql", "class", "script", "javascript")):
        return "ShareGPT/Coding"
    if any(k in low for k in ("server", "database", "latency", "docker", "api", "distributed", "cache", "system")):
        return "ShareGPT/SysDesign"
    if any(k in low for k in ("model", "neural", "matrix", "attention", "gradient", "probability", "math", "train")):
        return "ShareGPT/Math&ML"
    return "ShareGPT/Science"


def load_sharegpt_trace(
    num_requests: int = 12,
    qps: float = 2.0,
    max_new_tokens: int = 36,
    sharegpt_json_path: Optional[str] = None,
    download_from_hf: bool = False,
    seed: int = 42,
    stream_bytes: int = 1_500_000,
) -> List[TrafficRequest]:
    """
    Build a request trace with Poisson inter-arrival times.

    - If ``sharegpt_json_path`` points to a ShareGPT JSON file (or
      ``download_from_hf=True`` streams the first ``stream_bytes`` of the
      official ShareGPT dataset from HuggingFace), real human prompts are
      extracted.
    - Otherwise, uses ``BUILTIN_SHAREGPT_CONVERSATIONS`` so the monitor works
      instantly with zero extra downloads.

    Note:
        The trace **cycles** the available prompts when ``num_requests`` is
        larger than the number of unique prompts.  With prefix caching this is
        intentional: once every unique prefix is cached, KV usage plateaus at
        the unique-prefix footprint and later requests become cache hits.  Pass
        ``sharegpt_json_path`` (full dataset) or a larger ``stream_bytes`` for a
        workload with more unique prefixes.
    """
    rng = random.Random(seed)
    raw_items: List[Dict[str, str]] = []

    if download_from_hf and not sharegpt_json_path:
        try:
            import requests
            print("[ShareGPT] Streaming sample conversations from HuggingFace Hub...")
            resp = requests.get(SHAREGPT_HF_URL, stream=True, timeout=15)
            resp.raise_for_status()
            buf = b""
            for chunk in resp.iter_content(chunk_size=65536):
                buf += chunk
                if len(buf) >= stream_bytes:
                    break
            resp.close()
            text = buf.decode("utf-8", errors="ignore")
            # Parse complete conversation objects from the streamed prefix
            decoder = json.JSONDecoder()
            idx = text.find("{")
            while idx != -1 and len(raw_items) < num_requests * 3:
                try:
                    obj, end = decoder.raw_decode(text, idx)
                    convs = obj.get("conversations", [])
                    human = next(
                        (c["value"].strip() for c in convs if c.get("from") == "human" and 20 <= len(c.get("value", "")) <= 260),
                        None,
                    )
                    if human:
                        src = _classify_prompt(human)
                        raw_items.append({
                            "id": f"sgpt_{obj.get('id', len(raw_items))[:6]}",
                            "source": src,
                            "user": f"u_{abs(hash(obj.get('id', ''))) % 9000 + 1000}",
                            "prompt": " ".join(human.split()),
                        })
                    idx = text.find("{", end)
                except json.JSONDecodeError:
                    idx = text.find("{", idx + 1)
            print(f"[ShareGPT] Extracted {len(raw_items)} real human prompts from HuggingFace stream.")
        except Exception as exc:
            print(f"[ShareGPT] HF stream unavailable ({exc}); using built-in ShareGPT trace.")

    elif sharegpt_json_path and os.path.isfile(sharegpt_json_path):
        with open(sharegpt_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for obj in data:
            convs = obj.get("conversations", [])
            human = next(
                (c["value"].strip() for c in convs if c.get("from") == "human" and 20 <= len(c.get("value", "")) <= 260),
                None,
            )
            if human:
                raw_items.append({
                    "id": f"sgpt_{str(obj.get('id', len(raw_items)))[:6]}",
                    "source": _classify_prompt(human),
                    "user": f"u_{abs(hash(str(obj.get('id', '')))) % 9000 + 1000}",
                    "prompt": " ".join(human.split()),
                })
            if len(raw_items) >= num_requests * 2:
                break

    if not raw_items:
        raw_items = list(BUILTIN_SHAREGPT_CONVERSATIONS)

    # Keep one entry per unique prompt so the trace/replay diagnostics are exact.
    seen_prompts = set()
    unique_items = []
    for item in raw_items:
        if item["prompt"] not in seen_prompts:
            seen_prompts.add(item["prompt"])
            unique_items.append(item)
    raw_items = unique_items

    if num_requests > len(raw_items):
        print(
            f"[Trace] Only {len(raw_items)} unique prompts available; "
            f"{num_requests} requests will replay them. KV usage will plateau at the "
            f"unique-prefix footprint (prefix-cache steady state), not keep growing. "
            f"Pass sharegpt_json_path=<full ShareGPT.json> or a larger stream_bytes "
            f"for a unique-prompt load test."
        )

    # Build Poisson arrival schedule
    trace: List[TrafficRequest] = []
    t_cursor = 0.0
    for i in range(num_requests):
        item = raw_items[i % len(raw_items)]
        if i > 0 and qps > 0:
            t_cursor += rng.expovariate(qps)
        sys_prefix = CATEGORY_SYSTEM_PROMPTS.get(item["source"], "")
        trace.append(
            TrafficRequest(
                trace_id=f"{item['id']}#{i+1:02d}",
                source=item["source"],
                user=item["user"],
                user_prompt=item["prompt"],
                full_prompt=sys_prefix + "User: " + item["prompt"] + "\nAssistant:",
                arrival_offset_s=t_cursor,
                max_new_tokens=max_new_tokens,
            )
        )
    return trace


# ---------------------------------------------------------------------------
# Live Telemetry Dashboard (Colab HTML + Terminal ANSI/Rich)
# ---------------------------------------------------------------------------

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def _bar(frac: float, width: int = 22) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


class LiveTrafficMonitor:
    """
    Streams a :class:`TrafficRequest` workload into ``CustomInferenceEngine``
    while rendering a real-time dashboard on every engine step.
    """

    def __init__(
        self,
        engine,
        tokenizer=None,
        refresh_every_n_steps: int = 2,
        step_pause_s: float = 0.0,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.refresh_every_n_steps = max(1, refresh_every_n_steps)
        self.step_pause_s = step_pause_s

        # seq_id -> TrafficRequest metadata
        self.meta: Dict[str, TrafficRequest] = {}
        self.arrival_Feed: List[str] = []
        self.completed_feed: List[dict] = []
        self.source_counts: Dict[str, int] = {}

        # Prompt tokenization cache (traces often replay the same prompts)
        self._prompt_cache: Dict[str, List[int]] = {}

        # Load/telemetry high-water marks (instantaneous batch size can be 1
        # even when the scheduler is batching; peaks make bursts visible).
        self._unique_prompts = 0
        self.peak_waiting = 0
        self.peak_running = 0
        self.peak_prefill = 0
        self.peak_decode = 0
        self.peak_used_blocks = 0
        self._decode_batch_sum = 0
        self._decode_batch_samples = 0

    def _decode_tokens(self, token_ids: List[int]) -> str:
        if not token_ids:
            return ""
        if self.tokenizer is not None:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            text = " ".join(f"t{t}" for t in token_ids)
        return " ".join(text.split())

    def _encode_prompt(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        # Fallback deterministic word-hash tokenizer when running without HF
        words = text.split()
        return [(abs(hash(w)) % 200) + 10 for w in words]

    def _update_peaks(self) -> None:
        """Record high-water marks every engine step (not just render steps)."""
        sched = self.engine.scheduler
        bm = self.engine.block_manager
        lb = sched.last_batch
        self.peak_waiting = max(self.peak_waiting, len(sched.waiting_queue))
        self.peak_running = max(self.peak_running, len(sched.running_queue))
        self.peak_prefill = max(self.peak_prefill, len(lb.prefill_seqs))
        self.peak_decode = max(self.peak_decode, len(lb.decode_seqs))
        self.peak_used_blocks = max(
            self.peak_used_blocks, bm.num_blocks - len(bm.free_block_ids)
        )
        self._decode_batch_sum += len(lb.decode_seqs)
        self._decode_batch_samples += 1

    def run(
        self,
        trace: List[TrafficRequest],
        burst_initial: int = 3,
    ) -> List[dict]:
        """
        Execute the live traffic simulation until all requests in ``trace``
        have arrived and finished decoding.
        """
        self.engine.clear_prefix_cache()
        self.meta.clear()
        self.arrival_Feed.clear()
        self.completed_feed.clear()
        self.source_counts.clear()

        self._unique_prompts = len({req.full_prompt for req in trace})
        self.peak_waiting = 0
        self.peak_running = 0
        self.peak_prefill = 0
        self.peak_decode = 0
        self.peak_used_blocks = 0
        self._decode_batch_sum = 0
        self._decode_batch_samples = 0

        pending = list(trace)
        # Let the first `burst_initial` requests arrive immediately at t=0
        for j in range(min(burst_initial, len(pending))):
            pending[j].arrival_offset_s = 0.0

        use_nb = _in_notebook()
        display_handle = None
        if use_nb:
            from IPython.display import display, HTML
            display_handle = display(HTML("<pre>Starting Live Inference Monitor...</pre>"), display_id=True)

        t_start = time.perf_counter()
        step_idx = 0

        while pending or self.engine.has_unfinished_requests():
            elapsed = time.perf_counter() - t_start

            # 1. Admit any requests whose arrival timestamp has passed
            while pending and pending[0].arrival_offset_s <= elapsed:
                req = pending.pop(0)
                token_ids = self._prompt_cache.get(req.full_prompt)
                if token_ids is None:
                    token_ids = self._encode_prompt(req.full_prompt)
                    self._prompt_cache[req.full_prompt] = token_ids
                seq_id = self.engine.add_request(token_ids, max_tokens=req.max_new_tokens)
                self.meta[seq_id] = req
                self.source_counts[req.source] = self.source_counts.get(req.source, 0) + 1
                self.arrival_Feed.append(
                    f"[+{elapsed:5.2f}s] 📥 {req.trace_id} | {req.source:<18} | user={req.user} | "
                    f"prompt={len(token_ids)} tok | \"{req.user_prompt[:48]}...\""
                )

            # Record pre-schedule waiting queue depth before engine.step() drains it
            self.peak_waiting = max(self.peak_waiting, len(self.engine.scheduler.waiting_queue))

            # If engine is temporarily idle waiting for the next future arrival, advance clock
            if not self.engine.has_unfinished_requests() and pending:
                wait_s = max(0.0, pending[0].arrival_offset_s - (time.perf_counter() - t_start))
                if wait_s > 0:
                    time.sleep(min(wait_s, 0.25))
                    # Keep the dashboard visible while idle; otherwise low-QPS
                    # traces look frozen and then flash by in a single frame.
                    idle_now = time.perf_counter() - t_start
                    if use_nb and display_handle is not None:
                        from IPython.display import HTML
                        display_handle.update(
                            HTML(self._render_html(idle_now, len(trace), len(pending)))
                        )
                    else:
                        self._render_terminal(idle_now, len(trace), len(pending))
                continue

            # 2. Execute one iteration of Continuous Batching (Prefill + Decode)
            finished_Step = self.engine.step()
            step_idx += 1
            self._update_peaks()

            for out in finished_Step:
                req = self.meta.get(out.seq_id)
                self.completed_feed.append({
                    "seq_id": out.seq_id,
                    "trace_id": req.trace_id if req else out.seq_id,
                    "source": req.source if req else "custom",
                    "user": req.user if req else "-",
                    "prompt_len": out.prompt_len,
                    "cached_tokens": out.cached_tokens,
                    "output_len": out.output_len,
                    "ttft_ms": out.ttft_ms,
                    "tpot_ms": out.tpot_ms,
                    "text": self._decode_tokens(out.output_tokens),
                })

            # 3. Refresh live UI
            if (
                step_idx % self.refresh_every_n_steps == 0
                or finished_Step
                or not (pending or self.engine.has_unfinished_requests())
            ):
                elapsed_now = time.perf_counter() - t_start
                if use_nb and display_handle is not None:
                    from IPython.display import HTML
                    display_handle.update(
                        HTML(self._render_html(elapsed_now, len(trace), len(pending)))
                    )
                else:
                    self._render_terminal(elapsed_now, len(trace), len(pending))

            if self.step_pause_s > 0:
                time.sleep(self.step_pause_s)

        return self.completed_feed

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------
    def _collect_state(self, elapsed_s: float, total_reqs: int, pending_reqs: int) -> dict:
        sched = self.engine.scheduler
        bm = self.engine.block_manager
        last_batch = sched.last_batch

        prefill_ids = {s.seq_id for s in last_batch.prefill_seqs}
        decode_ids = {s.seq_id for s in last_batch.decode_seqs}

        used_blocks = bm.num_blocks - len(bm.free_block_ids)
        cached_blocks = bm.prefix_cache.num_cached_blocks
        gen_toks = sum(c["output_len"] for c in self.completed_feed) + sum(
            len(seq.output_tokens) for seq in sched.running_queue
        )
        tok_per_s = gen_toks / max(elapsed_s, 1e-3)
        avg_decode_batch = self._decode_batch_sum / max(self._decode_batch_samples, 1)
        repeat_factor = total_reqs / max(self._unique_prompts, 1)

        gpu_name = "CPU"
        vram_used_mb = 0.0
        vram_total_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available() and str(self.engine.device).startswith("cuda"):
                gpu_name = torch.cuda.get_device_name(0)
                vram_used_mb = torch.cuda.memory_allocated(0) / 1e6
                props = torch.cuda.get_device_properties(0)
                vram_total_mb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e6
        except Exception:
            pass

        active_rows = []
        for seq in list(sched.running_queue) + list(sched.waiting_queue):
            req = self.meta.get(seq.seq_id)
            phase = (
                "PREFILL ⚡" if seq.seq_id in prefill_ids
                else ("DECODE 🔄" if seq.seq_id in decode_ids else "QUEUED ⏳")
            )
            active_rows.append({
                "seq_id": seq.seq_id,
                "trace_id": req.trace_id if req else seq.seq_id,
                "source": req.source if req else "custom",
                "user": req.user if req else "-",
                "phase": phase,
                "prompt_len": len(seq.prompt_tokens),
                "cached": seq.cached_tokens,
                "gen_len": len(seq.output_tokens),
                "max_tokens": seq.max_tokens,
                "blocks": len(bm.get_block_table(seq.seq_id)),
                "question": req.user_prompt if req else "",
                "stream": self._decode_tokens(seq.output_tokens[-18:]),
            })

        return {
            "elapsed_s": elapsed_s,
            "step": self.engine.num_steps,
            "gpu_name": gpu_name,
            "vram_used_mb": vram_used_mb,
            "vram_total_mb": vram_total_mb,
            "total_reqs": total_reqs,
            "arrived_reqs": total_reqs - pending_reqs,
            "waiting_reqs": len(sched.waiting_queue),
            "running_reqs": len(sched.running_queue),
            "done_reqs": len(self.completed_feed),
            "prefill_batch": len(last_batch.prefill_seqs),
            "decode_batch": len(last_batch.decode_seqs),
            "used_blocks": used_blocks,
            "num_blocks": bm.num_blocks,
            "free_blocks": len(bm.free_block_ids),
            "cached_blocks": cached_blocks,
            "evicted_blocks": bm.num_evicted_blocks,
            "prefix_hit_pct": bm.prefix_hit_rate * 100.0,
            "cuda_replays": self.engine.cuda_graph_runner.num_replays,
            "tok_per_s": tok_per_s,
            "gen_tokens": gen_toks,
            "unique_prompts": self._unique_prompts,
            "repeat_factor": repeat_factor,
            "peak_waiting": self.peak_waiting,
            "peak_running": self.peak_running,
            "peak_prefill": self.peak_prefill,
            "peak_decode": self.peak_decode,
            "avg_decode_batch": avg_decode_batch,
            "peak_used_blocks": self.peak_used_blocks,
            "source_counts": dict(self.source_counts),
            "active_rows": active_rows,
            "arrivals": self.arrival_Feed[-6:],
            "completed": self.completed_feed[-10:],
        }

    # ------------------------------------------------------------------
    # Rich Dark-Mode HTML Terminal View (for Google Colab / Jupyter)
    # ------------------------------------------------------------------
    def _render_html(self, elapsed_s: float, total_reqs: int, pending_reqs: int) -> str:
        s = self._collect_state(elapsed_s, total_reqs, pending_reqs)
        kv_frac = s["used_blocks"] / max(s["num_blocks"], 1)
        req_frac = s["done_reqs"] / max(s["total_reqs"], 1)

        src_badges = " ".join(
            f"<span style='background:#1e293b;border:1px solid #334155;border-radius:4px;"
            f"padding:2px 8px;margin-right:6px;color:#38bdf8;font-size:11px;'>"
            f"{html.escape(k)}: <b>{v}</b></span>"
            for k, v in sorted(s["source_counts"].items())
        ) or "<span style='color:#64748b'>waiting...</span>"

        replay_note = ""
        if s["repeat_factor"] > 1.05:
            replay_note = (
                f"<div style='background:#1c1917;border:1px solid #78350f;color:#fbbf24;"
                f"padding:6px 10px;border-radius:6px;margin-bottom:8px;font-size:11px;'>"
                f"♻ Trace replays <b>{s['unique_prompts']}</b> unique prompts ×"
                f"<b>{s['repeat_factor']:.1f}</b>. Once every unique prefix is cached, KV usage "
                f"plateaus at the prefix footprint ({s['cached_blocks']} blocks) while hit rate "
                f"dominates — this is prefix-cache steady state, not a leak.</div>"
            )

        active_html = ""
        for r in s["active_rows"][:10]:
            p_color = "#f59e0b" if "PREFILL" in r["phase"] else ("#10b981" if "DECODE" in r["phase"] else "#94a3b8")
            prog = _bar(r["gen_len"] / max(r["max_tokens"], 1), 10)
            active_html += (
                f"<tr style='border-bottom:1px solid #1e293b;'>"
                f"<td style='padding:4px 6px;color:#e2e8f0;'>{html.escape(r['trace_id'])}</td>"
                f"<td style='padding:4px 6px;color:#38bdf8;'>{html.escape(r['source'])}</td>"
                f"<td style='padding:4px 6px;color:{p_color};font-weight:bold;'>{r['phase']}</td>"
                f"<td style='padding:4px 6px;color:#a7f3d0;'>{r['cached']}/{r['prompt_len']}</td>"
                f"<td style='padding:4px 6px;color:#cbd5e1;'>{prog} {r['gen_len']}/{r['max_tokens']}</td>"
                f"<td style='padding:4px 6px;color:#c084fc;'>{r['blocks']} blk</td>"
                f"<td style='padding:4px 6px;color:#fde68a;max-width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>"
                f"{html.escape(r['stream'] or r['question'][:50])}</td>"
                f"</tr>"
            )
        if not active_html:
            active_html = "<tr><td colspan='7' style='padding:8px;color:#64748b;text-align:center;'>All requests completed ✅</td></tr>"

        arr_html = "<br>".join(html.escape(line) for line in s["arrivals"]) or "—"
        done_html = ""
        for c in reversed(s["completed"]):
            done_html += (
                f"<div style='margin:2px 0;color:#cbd5e1;'>"
                f"✅ <b style='color:#4ade80'>{html.escape(c['trace_id'])}</b> "
                f"<span style='color:#38bdf8'>[{html.escape(c['source'])}]</span> "
                f"TTFT=<b>{c['ttft_ms']:.1f}ms</b> | TPOT=<b>{c['tpot_ms']:.1f}ms</b> | "
                f"prefix_hit={c['cached_tokens']}/{c['prompt_len']} tok → "
                f"<span style='color:#f8fafc'>\"{html.escape(c['text'][:68])}...\"</span></div>"
            )

        return f"""
<div style="background:#0f172a;color:#e2e8f0;font-family:'JetBrains Mono',Consolas,monospace;
            font-size:12px;padding:14px 16px;border-radius:10px;border:1px solid #334155;line-height:1.45;">
  <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #334155;padding-bottom:8px;margin-bottom:10px;">
    <div>
      <span style="color:#4ade80;font-weight:bold;font-size:14px;">⚡ CUSTOM INFERENCE ENGINE — LIVE GPU TELEMETRY</span>
      <span style="color:#94a3b8;margin-left:12px;">GPU: <b style="color:#f8fafc">{html.escape(s['gpu_name'])}</b>
      ({s['vram_used_mb']:.0f}/{s['vram_total_mb']:.0f} MB VRAM) |
      Unique prompts: <b style="color:#f8fafc">{s['unique_prompts']}</b>
      (replay ×{s['repeat_factor']:.1f})</span>
    </div>
    <div style="color:#38bdf8;">
      Step <b>#{s['step']}</b> | Elapsed <b>{s['elapsed_s']:.1f}s</b> |
      Gen Speed <b style="color:#4ade80;font-size:13px;">{s['tok_per_s']:.1f} tok/s</b>
      (<b>{s['gen_tokens']}</b> tok)
    </div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:10px;">
    <div style="background:#1e293b;padding:8px 10px;border-radius:6px;">
      <div style="color:#94a3b8;font-size:10px;">TRAFFIC &amp; QUEUES</div>
      <div style="font-size:13px;margin-top:2px;">
        Arrived: <b>{s['arrived_reqs']}/{s['total_reqs']}</b> |
        Wait: <b style="color:#facc15">{s['waiting_reqs']}</b><span style="color:#64748b">/{s['peak_waiting']}</span> |
        Run: <b style="color:#4ade80">{s['running_reqs']}</b><span style="color:#64748b">/{s['peak_running']}</span> |
        Done: <b style="color:#38bdf8">{s['done_reqs']}</b>
      </div>
      <div style="color:#64748b;margin-top:2px;">{_bar(req_frac, 20)} {req_frac*100:.0f}%</div>
    </div>

    <div style="background:#1e293b;padding:8px 10px;border-radius:6px;">
      <div style="color:#94a3b8;font-size:10px;">CONTINUOUS BATCHING (NOW / PEAK)</div>
      <div style="font-size:13px;margin-top:2px;">
        Prefill: <b style="color:#f59e0b">{s['prefill_batch']}</b><span style="color:#64748b">/{s['peak_prefill']}</span> seq |
        Decode: <b style="color:#10b981">{s['decode_batch']}</b><span style="color:#64748b">/{s['peak_decode']}</span> seq
      </div>
      <div style="color:#94a3b8;margin-top:2px;">Avg decode batch: <b>{s['avg_decode_batch']:.2f}</b> |
        CUDA Graph Replays: <b style="color:#c084fc">{s['cuda_replays']}</b></div>
    </div>

    <div style="background:#1e293b;padding:8px 10px;border-radius:6px;">
      <div style="color:#94a3b8;font-size:10px;">PAGED KV CACHE MEMORY</div>
      <div style="font-size:13px;margin-top:2px;">
        Blocks: <b>{s['used_blocks']}/{s['num_blocks']}</b> ({kv_frac*100:.1f}%)
        <span style="color:#64748b">peak {s['peak_used_blocks']/max(s['num_blocks'],1)*100:.1f}%</span>
      </div>
      <div style="color:#c084fc;margin-top:2px;">{_bar(kv_frac, 20)}</div>
      <div style="color:#94a3b8;margin-top:2px;">Free: <b>{s['free_blocks']}</b> |
        Evicted: <b>{s['evicted_blocks']}</b></div>
    </div>

    <div style="background:#1e293b;padding:8px 10px;border-radius:6px;">
      <div style="color:#94a3b8;font-size:10px;">RADIX PREFIX CACHE</div>
      <div style="font-size:13px;margin-top:2px;">
        Hit Rate: <b style="color:#4ade80">{s['prefix_hit_pct']:.1f}%</b>
      </div>
      <div style="color:#94a3b8;margin-top:2px;">Cached Blocks: <b>{s['cached_blocks']}</b> |
        Unique prompts: <b>{s['unique_prompts']}</b></div>
    </div>
  </div>

  {replay_note}

  <div style="margin-bottom:8px;">
    <span style="color:#94a3b8;font-size:11px;margin-right:8px;">🌐 INCOMING REQUEST SOURCES:</span>
    {src_badges}
  </div>

  <div style="background:#020617;padding:6px 10px;border-radius:6px;margin-bottom:10px;color:#94a3b8;font-size:11px;">
    <div style="color:#64748b;font-size:10px;margin-bottom:2px;">RECENT ARRIVALS (LIVE SHAREGPT FEED)</div>
    {arr_html}
  </div>

  <div style="margin-bottom:10px;">
    <div style="color:#94a3b8;font-size:10px;margin-bottom:4px;">ACTIVE CONTINUOUS-BATCHING SEQUENCES (LIVE TOKEN STREAM)</div>
    <table style="width:100%;border-collapse:collapse;font-size:11.5px;">
      <thead>
        <tr style="border-bottom:1px solid #334155;color:#94a3b8;text-align:left;">
          <th style="padding:3px 6px;">Req ID</th>
          <th style="padding:3px 6px;">Source</th>
          <th style="padding:3px 6px;">Stage</th>
          <th style="padding:3px 6px;">Prefix Hit</th>
          <th style="padding:3px 6px;">Decode Progress</th>
          <th style="padding:3px 6px;">KV Pages</th>
          <th style="padding:3px 6px;">Live Token Stream</th>
        </tr>
      </thead>
      <tbody>{active_html}</tbody>
    </table>
  </div>

  <div style="background:#020617;padding:6px 10px;border-radius:6px;font-size:11px;">
    <div style="color:#64748b;font-size:10px;margin-bottom:2px;">RECENTLY COMPLETED REQUESTS</div>
    {done_html or "<span style='color:#64748b'>No completed requests yet...</span>"}
  </div>
</div>
"""

    # ------------------------------------------------------------------
    # Plain Terminal Renderer (ANSI)
    # ------------------------------------------------------------------
    def _render_terminal(self, elapsed_s: float, total_reqs: int, pending_reqs: int) -> None:
        s = self._collect_state(elapsed_s, total_reqs, pending_reqs)
        kv_frac = s["used_blocks"] / max(s["num_blocks"], 1)
        lines = [
            "\033[2J\033[H" + "=" * 88,
            f" ⚡ CUSTOM INFERENCE ENGINE — LIVE TELEMETRY | GPU: {s['gpu_name']} | Step #{s['step']} | {s['elapsed_s']:.1f}s | {s['tok_per_s']:.1f} tok/s",
            "=" * 88,
            f" Requests : Arrived {s['arrived_reqs']}/{s['total_reqs']} | Waiting {s['waiting_reqs']}/{s['peak_waiting']} | Running {s['running_reqs']}/{s['peak_running']} (Prefill={s['prefill_batch']}/{s['peak_prefill']}, Decode={s['decode_batch']}/{s['peak_decode']}, avg={s['avg_decode_batch']:.2f}) | Done {s['done_reqs']}",
            f" KV Cache : [{_bar(kv_frac, 20)}] {s['used_blocks']}/{s['num_blocks']} blocks (peak {s['peak_used_blocks']/max(s['num_blocks'],1)*100:.1f}%) | Free={s['free_blocks']} | Evicted={s['evicted_blocks']} | Prefix Hit Rate: {s['prefix_hit_pct']:.1f}% ({s['cached_blocks']} blk) | CUDAGraphs: {s['cuda_replays']}",
            f" Trace    : {s['unique_prompts']} unique prompts | replay x{s['repeat_factor']:.1f} | Gen {s['gen_tokens']} tok"
            + ("  <-- KV plateau expected (prefix steady state)" if s["repeat_factor"] > 1.05 else ""),
            " Sources  : " + ", ".join(f"{k}={v}" for k, v in sorted(s["source_counts"].items())),
            "-" * 88,
            " RECENT ARRIVALS:",
        ]
        for a in s["arrivals"][-3:]:
            lines.append("   " + a)
        lines.append("-" * 88)
        lines.append(f" {'ID':<13} {'SOURCE':<18} {'STAGE':<11} {'PREFIX':<9} {'PROGRESS':<17} {'LIVE STREAM'}")
        for r in s["active_rows"][:8]:
            prog = f"{r['gen_len']}/{r['max_tokens']}"
            stream = (r["stream"] or r["question"])[:32]
            lines.append(
                f" {r['trace_id']:<13} {r['source']:<18} {r['phase']:<11} "
                f"{r['cached']}/{r['prompt_len']:<6} {prog:<17} {stream}"
            )
        if s["completed"]:
            lines.append("-" * 88)
            lines.append(" COMPLETED:")
            for c in s["completed"][-3:]:
                lines.append(
                    f"   ✅ {c['trace_id']} [{c['source']}] TTFT={c['ttft_ms']:.1f}ms TPOT={c['tpot_ms']:.1f}ms "
                    f"cached={c['cached_tokens']}/{c['prompt_len']} -> \"{c['text'][:45]}...\""
                )
        lines.append("=" * 88)
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

