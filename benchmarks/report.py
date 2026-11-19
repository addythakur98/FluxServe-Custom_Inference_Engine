"""Printing, plotting and JSON export of :class:`BenchmarkResult`."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Optional

from .harness import BenchmarkResult


def save_results_json(r: BenchmarkResult, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(r), f, indent=2, default=str)
    print(f"Results saved to {path}")
    return path


def _row(label: str, value: str) -> None:
    print(f"  {label:<52} {value:>24}")


def print_results(r: BenchmarkResult) -> None:
    e = r.env
    print("=" * 80)
    print(f"BENCHMARK RESULTS | {e.get('model_repo')} | {e.get('gpu') or e.get('device')} | {e.get('dtype')}")
    print("=" * 80)

    print("\n[Workload]")
    _row("Requests (N) / max new tokens", f"{r.num_requests} / {r.max_new_tokens}")
    _row("Prompt / output tokens (single-stream run)", f"{r.total_prompt_tokens} / {r.total_output_tokens}")

    print("\n[Single-stream latency, B=1]")
    _row("TTFT mean / P50 / P99 (ms)", f"{r.mean_ttft_ms:.2f} / {r.p50_ttft_ms:.2f} / {r.p99_ttft_ms:.2f}")
    _row("TPOT mean / P50 / P99 (ms)", f"{r.mean_tpot_ms:.2f} / {r.p50_tpot_ms:.2f} / {r.p99_tpot_ms:.2f}")
    _row(f"ITL P50 / P90 / P99 (ms), {r.num_itl_samples} samples",
         f"{r.p50_itl_ms:.2f} / {r.p90_itl_ms:.2f} / {r.p99_itl_ms:.2f}")
    _row("ITL jitter P99-P50 (ms)", f"{r.itl_jitter_ms:.2f}")

    print("\n[Throughput]")
    _row("Single-stream decode (tok/s)", f"{r.single_stream_decode_tps:.1f}")
    _row("Sequential B=1 serving, output (tok/s)", f"{r.sequential_output_tps:.1f}")
    _row("Continuous batching, output (tok/s)", f"{r.batched_output_tps:.1f}")
    _row("Continuous batching, prompt+output (tok/s)", f"{r.batched_total_tps:.1f}")
    _row("Batching speedup vs sequential", f"{r.batching_speedup:.2f}x")
    _row("Prefill-only, prefix cache off (tok/s)", f"{r.prefill_tps:.1f}")

    print("\n[Prefix caching]")
    _row("Token hit rate, sequential run (%)", f"{r.prefix_hit_rate_sequential_pct:.1f}")
    _row("Token hit rate, batched run (%)", f"{r.prefix_hit_rate_batched_pct:.1f}")
    _row("Token hit rate, shared-prefix cohort (%)", f"{r.prefix_cohort_hit_rate_pct:.1f}")
    _row("Cohort TTFT cold -> warm, mean (ms)",
         f"{r.prefix_ttft_cold_mean_ms:.2f} -> {r.prefix_ttft_warm_mean_ms:.2f}")
    _row("Cold vs warm outputs identical", "PASS" if r.prefix_exact_match else "FAIL")

    kv = r.kv_memory
    if kv:
        print("\n[KV-cache memory, time-weighted over the batched run]")
        _row("Waste: paged allocator (%)", f"{kv['paged_waste_pct']:.1f}")
        _row("Waste: static prompt+max_tokens reservation (%)", f"{kv['static_waste_pct']:.1f}")
        _row("Waste: static max_model_len reservation (%)", f"{kv['maxlen_waste_pct']:.1f}")
        _row("Waste reduction vs prompt+max_tokens (%)", f"{kv['waste_reduction_pct']:.1f}")
        _row("Waste reduction vs max_model_len (%)", f"{kv['waste_reduction_vs_maxlen_pct']:.1f}")
        _row("Reserved slots saved by prefix sharing (%)", f"{kv['prefix_sharing_savings_pct']:.1f}")

    ab = r.cuda_graph_ablation
    if ab is not None:
        print("\n[CUDA Graphs: off -> on]")
        _row("TPOT mean B=1 (ms)", f"{ab.eager_tpot_mean_ms:.2f} -> {ab.graph_tpot_mean_ms:.2f}")
        _row("ITL P50 / P99 B=1 (ms)",
             f"{ab.eager_itl_p50_ms:.1f}/{ab.eager_itl_p99_ms:.1f} -> {ab.graph_itl_p50_ms:.1f}/{ab.graph_itl_p99_ms:.1f}")
        _row("Host overhead per decode step B=1 (ms)",
             f"{ab.eager_cpu_overhead_ms:.2f} -> {ab.graph_cpu_overhead_ms:.2f}")
        _row("Single-stream (tok/s)", f"{ab.eager_single_tps:.1f} -> {ab.graph_single_tps:.1f}")
        _row("Batched (tok/s)", f"{ab.eager_batched_tps:.1f} -> {ab.graph_batched_tps:.1f}")

    print("\n[Memory & capacity]")
    _row("Weights / KV pool (MB)", f"{r.model_weight_memory_mb:.0f} / {r.kv_cache_memory_mb:.0f}")
    _row("Peak allocated / reserved (MB)", f"{r.peak_allocated_memory_mb:.0f} / {r.peak_reserved_memory_mb:.0f}")
    _row("Max concurrent seqs: unshared / shared / empirical",
         f"{r.max_concurrent_seqs_unshared} / {r.max_concurrent_seqs_shared} / {r.empirical_max_concurrent_seqs}")

    print("\n[Fidelity]")
    if r.greedy_match:
        _row("Engine vs full-recompute greedy tokens", f"{r.greedy_match['matched']}/{r.greedy_match['total']}")
    if r.perplexity:
        p = r.perplexity
        _row(f"WikiText-2 PPL, custom ({p.get('tokens', 0)} tokens)", f"{p['ppl']:.3f}")
        if "hf_ppl" in p:
            _row("WikiText-2 PPL, HF Transformers (same tokens)", f"{p['hf_ppl']:.3f}")

    for name, b in r.baselines.items():
        print(f"\n[Baseline: {name}]")
        _row("Single-stream (tok/s) / TPOT (ms)", f"{b['single_tps']:.1f} / {b['tpot_mean_ms']:.2f}")
        _row("Batched (tok/s)", f"{b['batched_tps']:.1f}")
    print("=" * 80)


def print_claims(r: BenchmarkResult) -> None:
    """The headline numbers, each with the definition it must be quoted with."""
    e, kv, ab = r.env, r.kv_memory or {}, r.cuda_graph_ablation
    hf = r.baselines.get("hf_transformers")
    print("=" * 80)
    print(f"HEADLINE NUMBERS ({e.get('model_repo')} on {e.get('gpu') or e.get('device')}, {e.get('dtype')})")
    print("=" * 80)
    lines = [
        f"Continuous batching: {r.batched_output_tps:.1f} output tok/s ({r.batched_total_tps:.1f} prompt+output tok/s) "
        f"vs {r.sequential_output_tps:.1f} tok/s serving the same {r.num_requests} requests one at a time "
        f"= {r.batching_speedup:.2f}x.",
        f"Decode ITL (B=1, CUDA graphs on): P50 {r.p50_itl_ms:.2f} ms, P99 {r.p99_itl_ms:.2f} ms "
        f"(jitter {r.itl_jitter_ms:.2f} ms) over {r.num_itl_samples} decode steps.",
    ]
    if ab is not None:
        lines.append(f"CUDA graphs: TPOT {ab.eager_tpot_mean_ms:.2f} -> {ab.graph_tpot_mean_ms:.2f} ms (B=1), "
                     f"host overhead {ab.eager_cpu_overhead_ms:.2f} -> {ab.graph_cpu_overhead_ms:.2f} ms/step.")
    if kv:
        lines.append(f"KV memory waste, paged (block {e.get('block_size')}): {kv['paged_waste_pct']:.1f}% vs "
                     f"{kv['maxlen_waste_pct']:.1f}% for static max_model_len={e.get('max_model_len')} reservation "
                     f"({kv['waste_reduction_vs_maxlen_pct']:.1f}% less) and {kv['static_waste_pct']:.1f}% for exact "
                     f"prompt+max_tokens reservation ({kv['waste_reduction_pct']:.1f}% less; negative = paged worse).")
    lines.append(f"Prefix cache: {r.prefix_hit_rate_sequential_pct:.1f}% token hit rate over the {r.num_requests}-request "
                 f"suite; shared-prefix cohort TTFT {r.prefix_ttft_cold_mean_ms:.1f} -> {r.prefix_ttft_warm_mean_ms:.1f} ms.")
    lines.append(f"Prefill (B=1, no prefix cache): {r.prefill_tps:.1f} tok/s.")
    if r.perplexity:
        p = r.perplexity
        s = f"WikiText-2 perplexity {p['ppl']:.3f} over {p['tokens']} tokens"
        if "hf_ppl" in p:
            s += f" (HF Transformers on identical tokens: {p['hf_ppl']:.3f})"
        lines.append(s + ".")
    if r.greedy_match:
        lines.append(f"Paged + graph decode matches full-recompute greedy decoding on "
                     f"{r.greedy_match['matched']}/{r.greedy_match['total']} tokens.")
    if hf:
        lines.append(f"vs HF Transformers: {r.batched_output_tps / max(hf['batched_tps'], 1e-9):.2f}x batched throughput "
                     f"({hf['batched_tps']:.1f} tok/s static batching).")
    for line in lines:
        print(" - " + line)
    print("=" * 80)


def plot_results(r: BenchmarkResult, path: Optional[str] = "benchmark_results.png") -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    fig.suptitle(f"{r.env.get('model_repo')} on {r.env.get('gpu') or r.env.get('device')} (N={r.num_requests})")

    ax = axes[0]
    if r.all_itls_ms:
        ax.hist(r.all_itls_ms, bins=40, color="#2a9d8f")
        for v, c, lbl in ((r.p50_itl_ms, "black", "P50"), (r.p99_itl_ms, "#e76f51", "P99")):
            ax.axvline(v, color=c, linestyle="--", label=f"{lbl} {v:.2f} ms")
        ax.legend()
    ax.set_title(f"Decode ITL, B=1 ({r.num_itl_samples} steps)")
    ax.set_xlabel("ms")

    ax = axes[1]
    labels, vals = ["Sequential\nB=1", "Continuous\nbatching"], [r.sequential_output_tps, r.batched_output_tps]
    for name, b in r.baselines.items():
        labels.append(f"{name}\nbatched")
        vals.append(b["batched_tps"])
    ax.bar(labels, vals, color="#264653")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    ax.set_title("Output throughput (tok/s)")

    ax = axes[2]
    kv = r.kv_memory or {}
    ax.bar(["Static\nmax_model_len", "Static\nprompt+max_tok", "Paged"],
           [kv.get("maxlen_waste_pct", 0), kv.get("static_waste_pct", 0), kv.get("paged_waste_pct", 0)],
           color=["#e76f51", "#f4a261", "#2a9d8f"])
    ax.set_title("KV-cache waste (% of reserved slots)")

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {path}")
    plt.show()
