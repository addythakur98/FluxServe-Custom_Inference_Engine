# Custom LLM Inference Engine

An LLM serving engine for dense Qwen2.5 checkpoints, written in PyTorch, implementing the core subsystems of **vLLM** / **SGLang**:

1. **PagedAttention KV cache**: block-based (block size 16) reference-counted KV allocator with per-sequence block tables; decode attention gathers K/V through the block table (pure PyTorch, CUDA-graph safe).
2. **Prefix caching**: block-granular radix tree with LRU leaf eviction, so requests sharing a prompt prefix reuse its KV blocks without recomputing them.
3. **Continuous batching**: iteration-level scheduler interleaving prefill and decode, with recompute preemption under KV pressure.
4. **CUDA Graph decode**: decode steps captured on a (batch size x context width) bucket grid and replayed to remove per-kernel launch overhead.
5. **Prefill attention**: PyTorch `scaled_dot_product_attention` (FlashAttention-2 backend on CUDA for plain causal prefill; prefill on top of a cached prefix uses an explicit mask).

---

## Project Structure

```text
custom_inference_engine/
├── .env                          # Single source of truth for model, device, KV & workload config
├── config.py                     # Centralized environment & settings loader (SETTINGS)
├── __init__.py                   # Top-level package exports
├── README.md                     # Architecture & usage guide
├── benchmarks/                   # Benchmark harness, workload, baselines & reporting
│   ├── harness.py                # BenchmarkHarness, perplexity, results dataclasses
│   ├── workload.py               # Stratified prompt suite with a shared-prefix cohort
│   ├── baselines.py              # HF Transformers / vLLM on the same GPU
│   └── report.py                 # Tables, headline numbers, plots, JSON export
├── core/                         # Core engine loop, scheduler & KV memory management
│   ├── __init__.py
│   ├── engine.py                 # CustomInferenceEngine & RequestOutput
│   ├── scheduler.py              # ContinuousScheduler, Sequence, ScheduleBatch
│   ├── block_manager.py          # BlockManager (Paged KV allocator)
│   └── prefix_cache.py           # PrefixCacheRadixTree, RadixNode, Block
├── kernels/                      # Attention & CUDA Graph execution kernels
│   ├── __init__.py
│   ├── flash_attention.py        # SDPA prefill attention
│   ├── paged_attention.py        # Vectorized PagedAttention decode
│   └── cuda_graph_runner.py      # Bucketed CUDA Graph capture & replay
├── models/                       # Model definition & weight loader
│   ├── __init__.py
│   ├── qwen2_model.py            # Qwen2Config & Qwen2Model (0.5B / 1.5B / 3B / 7B+)
│   └── weight_loader.py          # Single & multi-shard HuggingFace safetensors loader
├── telemetry/                    # Live traffic replay & terminal/Colab dashboard
│   ├── __init__.py
│   └── live_monitor.py           # LiveTrafficMonitor & ShareGPT trace loader
├── server/                       # Remote GPU HTTP API server & Colab tunnel launcher
│   ├── __init__.py
│   ├── engine_service.py         # Thread-owned continuous batching service + token streaming
│   ├── api_server.py             # FastAPI OpenAI-compatible server (/v1/chat/completions, SSE)
│   ├── colab_launcher.py         # Google Drive weight cache + uvicorn + Cloudflare/ngrok tunnel
│   └── requirements.txt          # Server-side Python dependencies
├── ui/                           # Standalone zero-build web UI (host anywhere)
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── notebooks/                    # Interactive Google Colab notebooks
│   ├── benchmark_colab.ipynb     # Colab benchmark: runs benchmarks/ and saves results JSON
│   └── colab_server.ipynb        # Colab A100 API server + Drive weight persistence + tunnel
└── scripts/                      # Validation & test runners
    ├── example_run.py            # End-to-end subsystem validation script
    ├── validate_nb.py            # Notebook structure & AST validator
    └── pack_for_colab.py         # Bundles engine code into dist/custom_inference_engine.zip
```

---

## Architecture Blueprint

```text
+-------------------------------------------------------------------------+
|                              Inference Engine                           |
+-------------------------------------------------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
|  Continuous Scheduler | <===================> |     Block Manager     |
| (Prefill vs. Decode)  |                       | (Paged KV Cache &     |
+-----------------------+                       |  Prefix Radix Tree)   |
            |                                   +-----------------------+
            |                                               |
            | Step Dispatch                                 | Block Tables
            v                                               v
+-------------------------------------------------------------------------+
|                             Model Worker                                |
|                                                                         |
|  [Prefill Phase]                                                        |
|     SDPA attention (FlashAttention-2 backend on CUDA)                   |
|     Populate Physical KV Cache Blocks                                   |
|                                                                         |
|  [Decode Phase] (CUDA Graph Replay)                                     |
|     PagedAttention (Block Tables -> Physical Slots)                     |
+-------------------------------------------------------------------------+
```

---

## Configuration (`.env`)

All model, device, KV cache, scheduler, and benchmark settings are configured in [`.env`](.env) and loaded automatically by [`config.py`](config.py):

```ini
ENGINE_MODEL_REPO=Qwen/Qwen2.5-7B-Instruct
ENGINE_MODEL_DIR=
ENGINE_DEVICE=auto
ENGINE_DTYPE=auto
ENGINE_NUM_BLOCKS=512
ENGINE_BLOCK_SIZE=16
ENGINE_MAX_BATCH_SIZE=16
```

---

## Quickstart

```bash
# 1. Run the validation suite
python scripts/example_run.py

# 2. Validate the Colab notebooks structure & AST
python scripts/validate_nb.py

# 3. Benchmark on Colab: pack the code (below), upload the zip to Drive,
#    open notebooks/benchmark_colab.ipynb on a GPU runtime and Run all
```

---

## Benchmarking

`notebooks/benchmark_colab.ipynb` runs `benchmarks/` on a real checkpoint and writes every number to
`MyDrive/inference_engine_results/<model>_<gpu>_<timestamp>.json` (plus a plot). `print_claims()` prints the
headline numbers together with how each was measured. Definitions:

| Metric | Definition |
|---|---|
| ITL P50 / P99, jitter | Inter-token latency of B=1 decode over all N requests, CUDA graphs on; jitter = P99 - P50 |
| Single-stream decode tok/s | (output tokens - N) / summed decode time, B=1 |
| Continuous-batching tok/s | Output (or prompt+output) tokens / wall time, all N requests submitted at once |
| Batching speedup | Continuous-batching output tok/s / sequential B=1 output tok/s on the same N requests |
| Prefill tok/s | Prompt tokens / prefill wall time, B=1, prefix cache off |
| Prefix hit rate | Prompt tokens served from cached blocks / prompt tokens (sequential run, batched run, shared-prefix cohort) |
| KV waste (paged vs static) | Time-weighted share of reserved KV slots not holding a live token, for the paged allocator vs static allocators reserving `max_model_len` or exactly prompt + max_tokens per request. Paging beats the first by a wide margin; against the second it only wins when requests stop early (EOS) |
| Perplexity | WikiText-2 (raw) test, non-overlapping 1024-token windows; the HF Transformers model is scored on the same tokens |
| Greedy match | Engine output (paged decode + CUDA graphs) vs a full-recompute greedy loop |
| Baselines | HF Transformers (B=1 and static batching) and optionally vLLM, same GPU / model / dtype |

---

## Results: Qwen2.5-7B-Instruct on NVIDIA A100-SXM4-40GB (fp16)

![Benchmark results](docs/benchmark_results.png)

100-request suite, 64 max new tokens, 512 KV blocks of 16 tokens, max batch 16. Raw data: [`docs/benchmark_results.json`](docs/benchmark_results.json).

| Metric | Result |
|---|---|
| Continuous batching, output throughput | **547.5 tok/s** (1505.0 prompt+output tok/s) |
| Same 100 requests served one at a time | 63.2 tok/s, so batching is **8.66x** faster |
| HF Transformers, static batching (same GPU) | 330.6 tok/s, so the engine is **1.66x** faster |
| Decode ITL, B=1 (6,300 steps) | P50 **15.38 ms**, P90 15.55 ms, P99 **15.60 ms** (jitter 0.23 ms) |
| CUDA graphs off → on, B=1 TPOT | 45.09 → **15.46 ms** (2.9x) |
| CUDA graphs off → on, batched throughput | 252.7 → 458.8 tok/s |
| TTFT, B=1 | mean 39.83 ms, P99 53.21 ms |
| Prefill-only throughput (B=1, prefix cache off) | 3114.5 tok/s |
| Prefix cache token hit rate | 65.2% (suite), 88.5% (shared-prefix cohort) |
| KV waste: paged / exact-length static / max-length static | **5.0%** / 18.5% / 93.0% |
| Reserved KV slots saved by prefix sharing | 33.7% |
| WikiText-2 perplexity (298,716 tokens) | **8.461**, identical to HF Transformers on the same tokens |
| Weights / KV pool / peak allocated | 15,231 MB / 471 MB / 16,017 MB |

**Known issues in this run**
- Decode correctness: the engine matched full-recompute greedy decoding on only 2/32 tokens. Decode attention computed Q·Kᵀ in fp16 before scaling, which overflows on Qwen2.5-7B and yields NaN logits. This is fixed in `kernels/paged_attention.py`, with a regression test in `scripts/example_run.py`, but the numbers above were measured **before** the fix, so decode outputs in this run were incorrect and decode timings may shift slightly on re-run. Perplexity uses the prefill path and is unaffected.
- Prefix caching cuts prompt compute (88.5% cohort hit rate), but the shared-prefix cohort's TTFT did not improve (37.74 → 42.12 ms): at ~340-token prompts the cached-context attention path costs more than the compute it saves.

---

## Remote Colab GPU Server + Standalone UI

Run the inference engine on a Colab A100 GPU with model weights cached permanently on Google Drive, while hosting the chat & telemetry UI locally or on any static host:

```bash
# 1. Pack engine source for Google Drive (creates dist/custom_inference_engine.zip)
python scripts/pack_for_colab.py

# 2. Upload dist/custom_inference_engine.zip to the root of your Google Drive (MyDrive/)
#    and open notebooks/colab_server.ipynb in Google Colab (A100 GPU) -> Run all cells.
#    It caches Qwen/Qwen2.5-Coder-0.5B-Instruct on Drive once and prints your API URL + API Key.

# 3. Serve the standalone UI locally (or deploy ui/ to Vercel / Netlify / GitHub Pages)
python -m http.server 5173 --directory ui
# Open http://localhost:5173 and paste the API URL & API Key into Settings.
```
