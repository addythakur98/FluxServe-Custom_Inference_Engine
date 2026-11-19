"""
Same-hardware, same-model, same-precision baselines: HuggingFace Transformers and vLLM.

Run these after releasing the custom engine (``engine.release_memory()``) so
each baseline has the GPU to itself.
"""

from __future__ import annotations

import gc
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .harness import perplexity


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_hf_baseline(
    model_dir: str,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    batch_size: int,
    dtype: torch.dtype,
    device: str = "cuda",
    max_model_len: int = 2048,
    ppl_token_ids: Optional[List[int]] = None,
    ppl_window: int = 1024,
    ppl_max_windows: Optional[int] = None,
) -> Dict[str, float]:
    """HF ``generate`` single-stream (first 16 prompts) and static-batched (all prompts), plus reference perplexity."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
    ).to(device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    res: Dict[str, float] = {}

    # Single stream (B=1)
    tpots, single_tokens = [], 0
    _sync()
    t_all = time.perf_counter()
    for p in tqdm(prompts[:16], desc="HF single-stream"):
        ids = tokenizer.encode(p, return_tensors="pt").to(device)
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        _sync()
        n = out.shape[1] - ids.shape[1]
        single_tokens += n
        if n > 1:
            tpots.append((time.perf_counter() - t0) * 1000.0 / n)
    res["single_tps"] = single_tokens / max(time.perf_counter() - t_all, 1e-9)
    res["tpot_mean_ms"] = float(np.mean(tpots)) if tpots else 0.0

    # Static batching (halves the batch on OOM)
    bs, idx, batched_tokens = batch_size, 0, 0
    _sync()
    t0 = time.perf_counter()
    while idx < len(prompts):
        chunk = prompts[idx: idx + bs]
        try:
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_model_len - max_new_tokens).to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            for row in out[:, enc.input_ids.shape[1]:].tolist():
                if tokenizer.eos_token_id in row:
                    row = row[: row.index(tokenizer.eos_token_id) + 1]
                batched_tokens += len(row)
            idx += len(chunk)
        except torch.cuda.OutOfMemoryError:
            _free()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
    _sync()
    res["batched_tps"] = batched_tokens / max(time.perf_counter() - t0, 1e-9)
    res["effective_batch_size"] = bs

    if ppl_token_ids is not None:
        def logits_fn(x, _m=model):
            return _m(x).logits
        res["ppl"] = perplexity(logits_fn, ppl_token_ids, device,
                                window=ppl_window, max_windows=ppl_max_windows)["ppl"]
        del logits_fn

    model.to("cpu")
    del model
    _free()
    return res


def run_vllm_baseline(
    model_dir: str,
    prompts: List[str],
    max_new_tokens: int,
    batch_size: int,
    dtype: torch.dtype,
    max_model_len: int = 2048,
) -> Optional[Dict[str, float]]:
    """vLLM on the identical workload; returns ``None`` if vLLM is unavailable."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("[vLLM] not installed (`pip install vllm` to include it); skipping.")
        return None
    try:
        vdtype = {torch.float16: "float16", torch.bfloat16: "bfloat16"}.get(dtype, "float32")
        llm = LLM(model=model_dir, dtype=vdtype, max_model_len=max_model_len,
                  max_num_seqs=batch_size, enable_prefix_caching=True, gpu_memory_utilization=0.80)
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        llm.generate(prompts[:4], sp, use_tqdm=False)    # warmup

        t0 = time.perf_counter()
        single = sum(len(llm.generate([p], sp, use_tqdm=False)[0].outputs[0].token_ids) for p in prompts[:16])
        single_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=True)
        batch_s = time.perf_counter() - t0
        batched = sum(len(o.outputs[0].token_ids) for o in outs)
        del llm
        _free()
        return {
            "single_tps": single / max(single_s, 1e-9),
            "tpot_mean_ms": single_s * 1000.0 / max(single, 1),
            "batched_tps": batched / max(batch_s, 1e-9),
        }
    except Exception as exc:  # vLLM version / GPU incompatibilities
        print(f"[vLLM] baseline skipped ({exc!r}).")
        return None
