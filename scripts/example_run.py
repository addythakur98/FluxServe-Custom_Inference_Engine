"""
Demonstration and validation script for the Custom Inference Engine.
Tests:
1. Prefix Caching (Radix Tree prefix match, block reuse & LRU eviction)
2. Paged KV Block Manager allocation & slot indexing
3. Continuous Batching Iteration Scheduler (prefill/decode interleaving & preemption)
4. Full End-to-End Engine (if PyTorch is installed)
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path when invoked directly as `python scripts/example_run.py`
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.block_manager import BlockManager  # noqa: E402
from core.scheduler import ContinuousScheduler, Sequence  # noqa: E402


def test_prefix_caching_and_blocks():
    print("\n--- [1] Testing Prefix Caching & Paged KV Cache ---")
    block_size = 16
    bm = BlockManager(num_blocks=64, block_size=block_size)

    system_prompt = [101, 202, 303, 404] * 4  # 16 tokens = 1 block
    user_query_1 = system_prompt + [501, 502, 503]
    user_query_2 = system_prompt + [601, 602, 603]

    print(f"Total Free Blocks initially: {len(bm.free_block_ids)}")

    # Request 1
    cached_tokens_1 = bm.allocate_for_prompt("req_1", user_query_1)
    print(f"Request 1: Cached tokens = {cached_tokens_1}, Assigned blocks = {bm.get_block_table('req_1')}")

    # Store request 1's system prompt block into Prefix Cache
    bm.cache_prefix("req_1", user_query_1)

    # Request 2 (shares the exact same system prompt)
    cached_tokens_2 = bm.allocate_for_prompt("req_2", user_query_2)
    print(f"Request 2: Cached tokens = {cached_tokens_2} (REUSED FROM CACHE!), Assigned blocks = {bm.get_block_table('req_2')}")

    # Verify both requests share the exact physical block
    shared_block = bm.get_block_table("req_1")[0]
    assert shared_block == bm.get_block_table("req_2")[0], "Prefix caching failed to share physical block!"
    print(f"SUCCESS: Physical block {shared_block} shared between req_1 and req_2 (ref_count={bm.blocks[shared_block].ref_count})")

    # Free both requests; cached block stays in radix tree (ref_count=1)
    bm.free_sequence("req_1")
    bm.free_sequence("req_2")
    assert bm.blocks[shared_block].ref_count == 1, "Radix tree should hold 1 ref on cached block"
    print(f"Prefix cache hit rate: {bm.prefix_hit_rate * 100:.1f}%")


def test_continuous_batching_scheduler():
    print("\n--- [2] Testing Continuous Batching Scheduler ---")
    bm = BlockManager(num_blocks=32, block_size=16)
    scheduler = ContinuousScheduler(block_manager=bm, max_batch_size=4)

    # Submit 2 requests
    scheduler.add_sequence(Sequence(seq_id="seq_A", prompt_tokens=[1, 2, 3, 4], max_tokens=3))
    scheduler.add_sequence(Sequence(seq_id="seq_B", prompt_tokens=[10, 20, 30], max_tokens=2))

    # Iteration 1: Prefill seq_A and seq_B
    step1 = scheduler.schedule()
    print(f"Iteration 1: Scheduled Prefills={[s.seq_id for s in step1.prefill_seqs]}, Decodes={[s.seq_id for s in step1.decode_seqs]}")
    for s in step1.prefill_seqs:
        s.output_tokens.append(999)

    # Iteration 2: Both sequences transition to Decode
    step2 = scheduler.schedule()
    print(f"Iteration 2: Scheduled Prefills={[s.seq_id for s in step2.prefill_seqs]}, Decodes={[s.seq_id for s in step2.decode_seqs]}")

    # Add a 3rd sequence while existing sequences are decoding (continuous batching!)
    scheduler.add_sequence(Sequence(seq_id="seq_C", prompt_tokens=[100, 200, 300], max_tokens=2))
    for s in step2.decode_seqs:
        s.output_tokens.append(999)

    # Iteration 3: seq_A decodes (seq_B finished at max_tokens=2) while seq_C prefills in the SAME step!
    step3 = scheduler.schedule()
    print(f"Iteration 3: Interleaved Prefills={[s.seq_id for s in step3.prefill_seqs]}, Decodes={[s.seq_id for s in step3.decode_seqs]}")
    assert [s.seq_id for s in step3.prefill_seqs] == ["seq_C"]
    assert [s.seq_id for s in step3.decode_seqs] == ["seq_A"]
    print("SUCCESS: Continuous batching interleaved prefill and decode in the same step.")


def test_end_to_end_engine_if_torch():
    try:
        import torch
    except ImportError:
        print("\n--- [3] Skipping PyTorch Model/Engine test (torch not installed locally) ---")
        return

    from models.qwen2_model import Qwen2Config, Qwen2Model
    from core.engine import CustomInferenceEngine

    print("\n--- [3] Testing End-to-End CustomInferenceEngine (Tiny Qwen2) ---")
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256, hidden_size=64, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128,
        num_hidden_layers=2, max_model_len=128, dtype=torch.float32,
    )
    model = Qwen2Model(cfg)
    engine = CustomInferenceEngine(model, num_blocks=32, block_size=8, use_cuda_graph=False)
    shared = list(range(10, 26))  # 16 tokens = 2 full blocks

    # Sequential requests so the second one actually observes the published
    # prefix blocks (a single generate() call admits both requests in the same
    # scheduler iteration, before either prefix is in the radix tree).
    out1 = engine.generate([shared + [1, 2]], max_tokens=6)[0]
    out2 = engine.generate([shared + [3, 4]], max_tokens=6)[0]
    for o in (out1, out2):
        print(f"  {o.seq_id}: cached={o.cached_tokens}, output={o.output_tokens}, TTFT={o.ttft_ms:.2f}ms")
    assert out2.cached_tokens == 16, "Second request should hit 2 cached blocks (16 tokens)"

    # Paged decode must agree with the full-sequence prefill reference.
    prompt = shared + [1, 2, 3]
    with torch.no_grad():
        ids = torch.tensor([prompt])
        for _ in range(6):
            pos = torch.arange(ids.shape[1]).unsqueeze(0)
            logits = model.forward_prefill(ids, pos, last_token_only=True)
            ids = torch.cat([ids, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
    ref = ids[0, len(prompt):].tolist()
    gen = engine.generate([prompt], max_tokens=6)[0].output_tokens
    assert gen == ref, f"Paged decode diverged from prefill reference: {gen} != {ref}"

    # KV accounting: a 3-token prompt with max_tokens=20 holds at most 23 tokens,
    # so the paged allocator (block_size 8) must waste less than static reservation.
    engine.reset_kv_accounting()
    engine.generate([[5, 6, 7], [8, 9, 10, 11]], max_tokens=20)
    kv = engine.kv_memory_stats()
    assert 0 <= kv["paged_waste_pct"] < kv["static_waste_pct"], kv
    print(f"  KV waste: static {kv['static_waste_pct']:.1f}% -> paged {kv['paged_waste_pct']:.1f}%")
    print("SUCCESS: End-to-end engine verified (prefix cache + paged decode correctness)!")


def test_paged_attention_fp16_large_activations():
    """fp16 decode attention must not overflow when |q.k| exceeds the fp16 range (Qwen2.5-7B regression)."""
    try:
        import torch
    except ImportError:
        return
    from kernels.paged_attention import paged_attention_decode

    print("\n--- [4] Testing PagedAttention fp16 with large activations ---")
    torch.manual_seed(0)
    heads, kv_heads, dim, bs, n_tok = 8, 2, 128, 16, 40
    q = torch.randn(1, heads, dim) * 100         # q.k std ~ 100*100*sqrt(128) >> 65504
    k = torch.randn(n_tok, kv_heads, dim) * 100
    v = torch.randn(n_tok, kv_heads, dim)
    k_cache = torch.zeros(4, bs, kv_heads, dim, dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    k_cache.view(-1, kv_heads, dim)[:n_tok] = k.half()
    v_cache.view(-1, kv_heads, dim)[:n_tok] = v.half()
    out = paged_attention_decode(q.half(), k_cache, v_cache,
                                 torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
                                 torch.tensor([n_tok], dtype=torch.int32), scale=dim ** -0.5).float()

    # Dense fp32 reference with GQA head expansion
    kf = k_cache.view(-1, kv_heads, dim)[:n_tok].float().repeat_interleave(heads // kv_heads, dim=1)
    vf = v_cache.view(-1, kv_heads, dim)[:n_tok].float().repeat_interleave(heads // kv_heads, dim=1)
    probs = torch.softmax(torch.einsum("hd,thd->ht", q.half().float()[0], kf) * dim ** -0.5, dim=-1)
    ref = torch.einsum("ht,thd->hd", probs, vf)
    assert torch.isfinite(out).all(), "PagedAttention produced inf/NaN (fp16 overflow)"
    err = (out[0] - ref).abs().max().item()
    assert err < 2e-2, f"PagedAttention fp16 mismatch vs fp32 reference: {err}"
    print(f"SUCCESS: fp16 PagedAttention finite and matches fp32 reference (max err {err:.1e}).")


if __name__ == "__main__":
    test_prefix_caching_and_blocks()
    test_continuous_batching_scheduler()
    test_end_to_end_engine_if_torch()
    test_paged_attention_fp16_large_activations()

    from telemetry.live_monitor import load_sharegpt_trace
    trace = load_sharegpt_trace(num_requests=6, qps=2.0, max_new_tokens=24, download_from_hf=False)
    print("\n--- [5] ShareGPT Traffic Trace Preview ---")
    for r in trace:
        print(f"  t=+{r.arrival_offset_s:4.2f}s | {r.trace_id} | {r.source:<18} | user={r.user} | {r.user_prompt[:55]}")

