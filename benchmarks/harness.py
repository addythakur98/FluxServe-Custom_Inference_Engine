"""
Benchmark harness for :class:`CustomInferenceEngine`.

Every number a write-up might quote is measured here and saved with
:func:`benchmarks.report.save_results_json`. Throughput figures are wall-clock
(``time.perf_counter`` around CUDA-synchronised work) unless the field name
says otherwise.
"""

from __future__ import annotations

import datetime as _dt
import math
import platform
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class RequestResult:
    """Metrics for a single generation request."""
    prompt_len: int = 0
    output_len: int = 0
    cached_tokens: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    itl_ms: List[float] = field(default_factory=list)
    total_time_ms: float = 0.0
    output_tokens: List[int] = field(default_factory=list)


@dataclass
class CUDAGraphAblationResult:
    """CUDA Graphs disabled (eager) vs enabled (graph replay) on an identical workload."""
    eager_tpot_mean_ms: float = 0.0
    eager_tpot_p99_ms: float = 0.0
    eager_single_tps: float = 0.0
    eager_gpu_util_pct: float = 0.0
    eager_cpu_overhead_ms: float = 0.0
    eager_cpu_overhead_pct: float = 0.0
    eager_itl_p50_ms: float = 0.0
    eager_itl_p99_ms: float = 0.0

    graph_tpot_mean_ms: float = 0.0
    graph_tpot_p99_ms: float = 0.0
    graph_single_tps: float = 0.0
    graph_gpu_util_pct: float = 0.0
    graph_cpu_overhead_ms: float = 0.0
    graph_cpu_overhead_pct: float = 0.0
    graph_itl_p50_ms: float = 0.0
    graph_itl_p99_ms: float = 0.0

    eager_batched_tps: float = 0.0
    eager_batched_step_ms: float = 0.0
    eager_batched_gpu_util_pct: float = 0.0
    eager_batched_cpu_overhead_ms: float = 0.0

    graph_batched_tps: float = 0.0
    graph_batched_step_ms: float = 0.0
    graph_batched_gpu_util_pct: float = 0.0
    graph_batched_cpu_overhead_ms: float = 0.0
    graph_replays: int = 0


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    env: Dict[str, object] = field(default_factory=dict)

    num_requests: int = 0
    max_new_tokens: int = 0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0

    # --- Single-stream (B=1, requests run one after another) latency ---
    mean_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    p50_tpot_ms: float = 0.0
    p90_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    num_itl_samples: int = 0
    mean_itl_ms: float = 0.0
    p50_itl_ms: float = 0.0
    p90_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    itl_jitter_ms: float = 0.0              # P99 - P50 ITL

    # --- Throughput ---
    # B=1 decode rate: generated tokens after the first / summed decode time.
    single_stream_decode_tps: float = 0.0
    # Sequential B=1 serving of all N requests: output tokens / wall time.
    sequential_output_tps: float = 0.0
    # Continuous batching of the same N requests: output (and prompt+output) tokens / wall time.
    batched_output_tps: float = 0.0
    batched_total_tps: float = 0.0
    batched_wall_s: float = 0.0
    batching_speedup: float = 0.0           # batched_output_tps / sequential_output_tps
    # Prefill-only rate: prompt tokens / prefill wall time (B=1, prefix cache OFF).
    prefill_tps: float = 0.0

    # --- Prefix caching ---
    prefix_hit_rate_sequential_pct: float = 0.0   # token hit rate, sequential run over all N
    prefix_hit_rate_batched_pct: float = 0.0      # token hit rate, continuous-batching run
    prefix_cohort_hit_rate_pct: float = 0.0       # hit rate within the shared-prefix cohort
    prefix_ttft_cold_mean_ms: float = 0.0
    prefix_ttft_cold_p99_ms: float = 0.0
    prefix_ttft_warm_mean_ms: float = 0.0
    prefix_ttft_warm_p99_ms: float = 0.0
    prefix_exact_match: bool = True
    cuda_graph_replays: int = 0

    # --- KV-cache memory (time-weighted over the continuous-batching run) ---
    kv_memory: Dict[str, float] = field(default_factory=dict)

    # --- Memory & capacity ---
    model_weight_memory_mb: float = 0.0
    kv_cache_memory_mb: float = 0.0
    peak_allocated_memory_mb: float = 0.0
    peak_reserved_memory_mb: float = 0.0
    max_concurrent_seqs_unshared: int = 0
    max_concurrent_seqs_shared: int = 0
    empirical_max_concurrent_seqs: int = 0

    # --- Quality / fidelity ---
    perplexity: Optional[Dict[str, float]] = None
    greedy_match: Optional[Dict[str, int]] = None

    cuda_graph_ablation: Optional[CUDAGraphAblationResult] = None
    baselines: Dict[str, Dict[str, float]] = field(default_factory=dict)

    all_itls_ms: List[float] = field(default_factory=list)
    all_ttfts_ms: List[float] = field(default_factory=list)


def _pct(values, q) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


def collect_env(engine, settings=None) -> Dict[str, object]:
    """Hardware / software / config fingerprint stored alongside the results."""
    env: Dict[str, object] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(engine.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dtype": str(engine.dtype).replace("torch.", ""),
        "num_parameters": sum(p.numel() for p in engine.model.parameters()),
        "num_blocks": engine.num_blocks,
        "block_size": engine.block_size,
        "max_batch_size": engine.scheduler.max_batch_size,
        "max_model_len": engine.max_model_len,
        "cuda_graph_captured": engine.cuda_graph_runner.is_captured,
    }
    if settings is not None:
        env["model_repo"] = settings.model_repo
    return env


def wikitext2_token_ids(tokenizer) -> List[int]:
    """Token ids of the WikiText-2 (raw) test split, joined as in the standard HF recipe."""
    from datasets import load_dataset

    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return tokenizer("\n\n".join(test["text"]))["input_ids"]


def perplexity(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    token_ids: List[int],
    device,
    window: int = 1024,
    max_windows: Optional[int] = None,
) -> Dict[str, float]:
    """
    Perplexity over non-overlapping ``window``-token chunks.

    ``logits_fn(input_ids[1, T]) -> logits[1, T, vocab]``. Using the same
    function shape for the custom model and the HF reference lets the two be
    compared on exactly the same tokens.
    """
    n_windows = len(token_ids) // window
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    nll, count = 0.0, 0
    with torch.no_grad():
        for w in tqdm(range(n_windows), desc="Perplexity"):
            chunk = torch.tensor([token_ids[w * window:(w + 1) * window]], dtype=torch.long, device=device)
            logits = logits_fn(chunk)[:, :-1, :].float()
            nll += torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1), reduction="sum"
            ).item()
            count += window - 1
            del logits
    return {"ppl": math.exp(nll / max(count, 1)), "tokens": count, "windows": n_windows, "window": window}


class BenchmarkHarness:
    """
    1. Single-stream latency over N requests (TTFT / TPOT / ITL distribution)
    2. Prefix-cache validation (cold vs warm TTFT + exact token equivalence)
    3. Continuous-batching throughput on the same N requests + KV memory accounting
    4. Prefill-only throughput (prefix cache off)
    5. CUDA Graphs disabled vs enabled ablation
    6. Memory breakdown & concurrency limits
    """

    def __init__(self, engine, tokenizer, settings=None):
        self.engine = engine
        self.model = engine.model
        self.tokenizer = tokenizer
        self.settings = settings
        self.is_cuda = engine.device.type == "cuda"

    def _sync(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize()

    def _generate_single(self, prompt: str, max_new_tokens: int) -> RequestResult:
        input_ids = self.tokenizer.encode(prompt)
        self._sync()
        out = self.engine.generate([input_ids], max_tokens=max_new_tokens)[0]
        self._sync()
        return RequestResult(
            prompt_len=out.prompt_len,
            output_len=out.output_len,
            cached_tokens=out.cached_tokens,
            ttft_ms=out.ttft_ms,
            tpot_ms=out.tpot_ms,
            itl_ms=out.itl_ms,
            total_time_ms=out.e2e_ms,
            output_tokens=out.output_tokens,
        )

    # ------------------------------------------------------------------
    def validate_prefix_caching(self, prefix_prompts: List[str], max_new_tokens: int = 16) -> dict:
        """Cold (cache cleared every request) vs warm (shared prefix cached) TTFT + token equivalence."""
        bm = self.engine.block_manager
        cold_ttfts, cold_tokens = [], []
        bm.enable_prefix_caching = False
        for p in tqdm(prefix_prompts, desc="Prefix cache COLD"):
            self.engine.clear_prefix_cache()
            res = self._generate_single(p, max_new_tokens)
            cold_ttfts.append(res.ttft_ms)
            cold_tokens.append(res.output_tokens)

        bm.enable_prefix_caching = True
        self.engine.clear_prefix_cache()
        prime = self._generate_single(prefix_prompts[0], max_new_tokens)
        warm_ttfts, warm_tokens = [], [prime.output_tokens]
        cached, total = 0, 0
        for p in tqdm(prefix_prompts[1:], desc="Prefix cache WARM"):
            res = self._generate_single(p, max_new_tokens)
            warm_ttfts.append(res.ttft_ms)
            warm_tokens.append(res.output_tokens)
            cached += res.cached_tokens
            total += res.prompt_len

        return {
            "cold_mean_ms": float(np.mean(cold_ttfts)),
            "cold_p99_ms": _pct(cold_ttfts, 99),
            "warm_mean_ms": float(np.mean(warm_ttfts)) if warm_ttfts else prime.ttft_ms,
            "warm_p99_ms": _pct(warm_ttfts, 99) if warm_ttfts else prime.ttft_ms,
            "cohort_hit_rate_pct": 100.0 * cached / max(total, 1),
            "exact_match": all(c == w for c, w in zip(cold_tokens, warm_tokens)),
        }

    # ------------------------------------------------------------------
    def measure_prefill_throughput(self, prompts: List[str]) -> float:
        """Prompt tokens / prefill wall time, one request at a time, prefix cache OFF."""
        bm = self.engine.block_manager
        prev = bm.enable_prefix_caching
        bm.enable_prefix_caching = False
        self.engine.clear_prefix_cache()
        tokens, elapsed = 0, 0.0
        try:
            for p in tqdm(prompts, desc="Prefill throughput"):
                ids = self.tokenizer.encode(p)
                self._sync()
                t0 = time.perf_counter()
                self.engine.generate([ids], max_tokens=1)   # exactly one prefill step
                self._sync()
                elapsed += time.perf_counter() - t0
                tokens += len(ids)
        finally:
            bm.enable_prefix_caching = prev
        return tokens / max(elapsed, 1e-9)

    # ------------------------------------------------------------------
    def run_cuda_graph_ablation(self, prompts: List[str], max_new_tokens: int = 32) -> CUDAGraphAblationResult:
        """CUDA Graphs disabled vs enabled. CUDA events separate GPU time from host launch overhead."""
        ablation = CUDAGraphAblationResult()
        encoded_single = [self.tokenizer.encode(p) for p in prompts[:16]]
        encoded_batch = [self.tokenizer.encode(p) for p in prompts[: max(16, self.engine.scheduler.max_batch_size)]]
        use_events = self.is_cuda

        def timed_step():
            if use_events:
                s_evt = torch.cuda.Event(enable_timing=True)
                e_evt = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                s_evt.record()
                fin = self.engine.step()
                e_evt.record()
                torch.cuda.synchronize()
                wall = (time.perf_counter() - t0) * 1000.0
                return fin, wall, min(float(s_evt.elapsed_time(e_evt)), wall)
            t0 = time.perf_counter()
            fin = self.engine.step()
            wall = (time.perf_counter() - t0) * 1000.0
            return fin, wall, wall

        def profile(enable_graph: bool) -> dict:
            self.engine.set_cuda_graph_enabled(enable_graph)
            self.engine.clear_prefix_cache()
            self.engine.generate([encoded_single[0]], max_tokens=4)   # warmup
            self.engine.clear_prefix_cache()
            self.engine.cuda_graph_runner.num_replays = 0
            self.engine.cuda_graph_runner.num_eager = 0

            # A. Single stream (B=1)
            tpots, itls, walls, gpus = [], [], [], []
            gen = 0
            self._sync()
            t_start = time.perf_counter()
            for ids in encoded_single:
                self.engine.clear_prefix_cache()
                self.engine.add_request(ids, max_tokens=max_new_tokens)
                finished = self.engine.step()            # prefill
                self._sync()
                while self.engine.has_unfinished_requests():
                    finished, wall, gpu = timed_step()
                    walls.append(wall)
                    gpus.append(gpu)
                for out in finished:
                    gen += out.output_len
                    itls.extend(out.itl_ms)
                    if out.tpot_ms > 0:
                        tpots.append(out.tpot_ms)
            single_tps = gen / max(time.perf_counter() - t_start, 1e-9)
            mean_wall = float(np.mean(walls)) if walls else 1.0
            mean_gpu = float(np.mean(gpus)) if gpus else mean_wall
            gpu_util = min(100.0, 100.0 * mean_gpu / max(mean_wall, 1e-9))

            # B. Continuous batching (all prompts submitted at once)
            self.engine.clear_prefix_cache()
            for ids in encoded_batch:
                self.engine.add_request(ids, max_tokens=max_new_tokens)
            b_walls, b_gpus, outs = [], [], []
            self._sync()
            t_batch = time.perf_counter()
            while self.engine.has_unfinished_requests():
                if not self.engine.scheduler.waiting_queue:     # decode-only step
                    fin, wall, gpu = timed_step()
                    b_walls.append(wall)
                    b_gpus.append(gpu)
                else:
                    fin = self.engine.step()
                outs.extend(fin)
            self._sync()
            batch_tps = sum(o.output_len for o in outs) / max(time.perf_counter() - t_batch, 1e-9)
            b_wall = float(np.mean(b_walls)) if b_walls else 1.0
            b_gpu = float(np.mean(b_gpus)) if b_gpus else b_wall

            return {
                "tpot_mean": float(np.mean(tpots)) if tpots else mean_wall,
                "tpot_p99": _pct(tpots, 99) if tpots else mean_wall,
                "itl_p50": _pct(itls, 50),
                "itl_p99": _pct(itls, 99),
                "single_tps": single_tps,
                "gpu_util_pct": gpu_util,
                "cpu_overhead_ms": max(0.0, mean_wall - mean_gpu),
                "cpu_overhead_pct": max(0.0, 100.0 - gpu_util),
                "batched_tps": batch_tps,
                "batched_step_ms": b_wall,
                "batched_gpu_util_pct": min(100.0, 100.0 * b_gpu / max(b_wall, 1e-9)),
                "batched_cpu_overhead_ms": max(0.0, b_wall - b_gpu),
                "replays": self.engine.cuda_graph_runner.num_replays,
            }

        print("\n--- [Ablation 1/2] CUDA Graphs DISABLED (eager decode) ---")
        eager = profile(False)
        print("--- [Ablation 2/2] CUDA Graphs ENABLED (graph replay) ---")
        graph = profile(True)

        for prefix, d in (("eager", eager), ("graph", graph)):
            setattr(ablation, f"{prefix}_tpot_mean_ms", d["tpot_mean"])
            setattr(ablation, f"{prefix}_tpot_p99_ms", d["tpot_p99"])
            setattr(ablation, f"{prefix}_itl_p50_ms", d["itl_p50"])
            setattr(ablation, f"{prefix}_itl_p99_ms", d["itl_p99"])
            setattr(ablation, f"{prefix}_single_tps", d["single_tps"])
            setattr(ablation, f"{prefix}_gpu_util_pct", d["gpu_util_pct"])
            setattr(ablation, f"{prefix}_cpu_overhead_ms", d["cpu_overhead_ms"])
            setattr(ablation, f"{prefix}_cpu_overhead_pct", d["cpu_overhead_pct"])
            setattr(ablation, f"{prefix}_batched_tps", d["batched_tps"])
            setattr(ablation, f"{prefix}_batched_step_ms", d["batched_step_ms"])
            setattr(ablation, f"{prefix}_batched_gpu_util_pct", d["batched_gpu_util_pct"])
            setattr(ablation, f"{prefix}_batched_cpu_overhead_ms", d["batched_cpu_overhead_ms"])
        ablation.graph_replays = graph["replays"]
        return ablation

    # ------------------------------------------------------------------
    def measure_empirical_concurrency_limit(self, prompt_len: int = 128, max_new_tokens: int = 64) -> int:
        """Concurrent sequences the KV pool sustains before the scheduler first preempts (scheduler-only simulation)."""
        from core.block_manager import BlockManager
        from core.scheduler import ContinuousScheduler, Sequence

        bm = BlockManager(num_blocks=self.engine.num_blocks, block_size=self.engine.block_size,
                          enable_prefix_caching=False)
        sched = ContinuousScheduler(block_manager=bm, max_batch_size=self.engine.num_blocks,
                                    max_batched_tokens=self.engine.num_blocks * self.engine.block_size)
        for i in range(self.engine.num_blocks + 1):
            dummy = [(i * 31 + j) % 1000 + 10 for j in range(prompt_len)]
            sched.add_sequence(Sequence(seq_id=f"probe_{i}", prompt_tokens=dummy, max_tokens=max_new_tokens))

        batch = sched.schedule()
        for s in batch.prefill_seqs:
            s.output_tokens.append(42)
        sustained = len(sched.running_queue)
        for _ in range(max_new_tokens - 1):
            batch = sched.schedule()
            if any(s.num_preemptions for s in sched.running_queue + sched.waiting_queue):
                return len(batch.decode_seqs)
            for s in batch.decode_seqs:
                s.output_tokens.append(42)
        return sustained

    # ------------------------------------------------------------------
    def greedy_match_vs_prefill(self, prompt: str, max_new_tokens: int = 32) -> Dict[str, int]:
        """Engine output (paged decode + CUDA graphs) vs a full-recompute greedy loop: identical tokens?"""
        ids = self.tokenizer.encode(prompt)
        self.engine.clear_prefix_cache()
        gen = self.engine.generate([ids], max_tokens=max_new_tokens)[0].output_tokens
        with torch.no_grad():
            ref = torch.tensor([ids], dtype=torch.long, device=self.engine.device)
            for _ in range(len(gen)):
                pos = torch.arange(ref.shape[1], device=self.engine.device).unsqueeze(0)
                nxt = self.model.forward_prefill(ref, pos, last_token_only=True)[:, -1, :].argmax(-1, keepdim=True)
                ref = torch.cat([ref, nxt], dim=1)
        ref_tokens = ref[0, len(ids):].tolist()
        return {"matched": sum(a == b for a, b in zip(gen, ref_tokens)), "total": len(ref_tokens)}

    def custom_logits_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def fn(input_ids: torch.Tensor) -> torch.Tensor:
            pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            return self.model.forward_prefill(input_ids, pos)
        return fn

    # ------------------------------------------------------------------
    def run(self, prompts: List[str], max_new_tokens: int = 64, shared_prefix: str = "") -> BenchmarkResult:
        eng = self.engine
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()

        # Warmup
        eng.set_cuda_graph_enabled(True)
        eng.clear_prefix_cache()
        self._generate_single("Warmup request.", 8)
        eng.generate([self.tokenizer.encode(p) for p in prompts[:16]], max_tokens=4)
        eng.clear_prefix_cache()
        eng.cuda_graph_runner.num_replays = 0
        eng.cuda_graph_runner.num_eager = 0

        # 1. Single-stream latency (sequential, B=1)
        results: List[RequestResult] = []
        self._sync()
        t_seq = time.perf_counter()
        for p in tqdm(prompts, desc=f"Single-stream latency (N={len(prompts)})"):
            results.append(self._generate_single(p, max_new_tokens))
        seq_wall = time.perf_counter() - t_seq

        br = self._aggregate(results)
        br.env = collect_env(eng, self.settings)
        br.max_new_tokens = max_new_tokens
        br.sequential_output_tps = br.total_output_tokens / max(seq_wall, 1e-9)
        br.cuda_graph_replays = eng.cuda_graph_runner.num_replays
        br.prefix_hit_rate_sequential_pct = eng.block_manager.prefix_hit_rate * 100.0

        # 2. Prefix-cache validation
        cohort = [p for p in prompts if shared_prefix and p.startswith(shared_prefix[:40])]
        if len(cohort) >= 2:
            pc = self.validate_prefix_caching(cohort, max_new_tokens=16)
            br.prefix_ttft_cold_mean_ms = pc["cold_mean_ms"]
            br.prefix_ttft_cold_p99_ms = pc["cold_p99_ms"]
            br.prefix_ttft_warm_mean_ms = pc["warm_mean_ms"]
            br.prefix_ttft_warm_p99_ms = pc["warm_p99_ms"]
            br.prefix_cohort_hit_rate_pct = pc["cohort_hit_rate_pct"]
            br.prefix_exact_match = pc["exact_match"]

        # 3. Continuous batching on the same N requests + KV memory accounting
        eng.clear_prefix_cache()
        eng.reset_kv_accounting()
        encoded = [self.tokenizer.encode(p) for p in prompts]
        self._sync()
        t0 = time.perf_counter()
        outs = eng.generate(encoded, max_tokens=max_new_tokens)
        self._sync()
        br.batched_wall_s = time.perf_counter() - t0
        gen = sum(o.output_len for o in outs)
        prompt_toks = sum(o.prompt_len for o in outs)
        br.batched_output_tps = gen / max(br.batched_wall_s, 1e-9)
        br.batched_total_tps = (gen + prompt_toks) / max(br.batched_wall_s, 1e-9)
        br.batching_speedup = br.batched_output_tps / max(br.sequential_output_tps, 1e-9)
        br.prefix_hit_rate_batched_pct = eng.block_manager.prefix_hit_rate * 100.0
        br.kv_memory = eng.kv_memory_stats()

        # 4. Prefill-only throughput
        br.prefill_tps = self.measure_prefill_throughput(prompts)

        # 5. CUDA Graph ablation
        br.cuda_graph_ablation = self.run_cuda_graph_ablation(prompts, max_new_tokens=min(max_new_tokens, 32))

        # 6. Memory & capacity
        avg_prompt = int(round(br.total_prompt_tokens / max(br.num_requests, 1)))
        shared_toks = len(self.tokenizer.encode(shared_prefix)) if shared_prefix else 0
        mem = eng.get_memory_profile(avg_prompt_len=avg_prompt, max_new_tokens=max_new_tokens,
                                     shared_prefix_tokens=shared_toks)
        br.model_weight_memory_mb = mem["model_weight_memory_mb"]
        br.kv_cache_memory_mb = mem["kv_cache_memory_mb"]
        br.peak_allocated_memory_mb = mem["peak_allocated_memory_mb"]
        br.peak_reserved_memory_mb = mem["peak_reserved_memory_mb"]
        br.max_concurrent_seqs_unshared = int(mem["max_concurrent_seqs_unshared"])
        br.max_concurrent_seqs_shared = int(mem["max_concurrent_seqs_prefix_shared"])
        br.empirical_max_concurrent_seqs = self.measure_empirical_concurrency_limit(avg_prompt, max_new_tokens)

        # 7. Decode fidelity
        br.greedy_match = self.greedy_match_vs_prefill("The capital of France is", max_new_tokens=32)
        eng.set_cuda_graph_enabled(True)
        return br

    @staticmethod
    def _aggregate(results: List[RequestResult]) -> BenchmarkResult:
        br = BenchmarkResult()
        br.num_requests = len(results)
        ttfts = [r.ttft_ms for r in results]
        tpots = [r.tpot_ms for r in results if r.tpot_ms > 0]
        itls = [x for r in results for x in r.itl_ms]
        br.all_itls_ms, br.all_ttfts_ms = itls, ttfts
        br.total_prompt_tokens = sum(r.prompt_len for r in results)
        br.total_output_tokens = sum(r.output_len for r in results)

        br.mean_ttft_ms = float(np.mean(ttfts))
        br.p50_ttft_ms, br.p90_ttft_ms, br.p99_ttft_ms = (_pct(ttfts, q) for q in (50, 90, 99))
        if tpots:
            br.mean_tpot_ms = float(np.mean(tpots))
            br.p50_tpot_ms, br.p90_tpot_ms, br.p99_tpot_ms = (_pct(tpots, q) for q in (50, 90, 99))
        if itls:
            br.num_itl_samples = len(itls)
            br.mean_itl_ms = float(np.mean(itls))
            br.p50_itl_ms, br.p90_itl_ms, br.p99_itl_ms = (_pct(itls, q) for q in (50, 90, 99))
            br.itl_jitter_ms = br.p99_itl_ms - br.p50_itl_ms
            # Requests ran one at a time, so summed ITLs == decode wall time.
            br.single_stream_decode_tps = len(itls) / (sum(itls) / 1000.0)
        return br
