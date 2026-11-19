"""PagedAttention decode: attention over a block-table-indexed (paged) KV cache."""

import math
from typing import Optional

import torch


def paged_attention_decode(
    query: "torch.Tensor",            # [batch_size, num_heads, head_dim]
    key_cache: "torch.Tensor",        # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: "torch.Tensor",      # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: "torch.Tensor",     # [batch_size, max_blocks_per_seq]
    seq_lens: "torch.Tensor",         # [batch_size]
    scale: Optional[float] = None,
) -> "torch.Tensor":
    """
    Vectorized PagedAttention decode (pure PyTorch, CUDA-Graph safe).

    Performs the block-table gather for the whole batch at once with *static shapes*
    and no host synchronisation (no ``.item()``), so it can be captured into
    a CUDA Graph. Positions beyond each sequence's length are masked out.

    Every ``seq_lens`` entry must be >= 1 (padding rows should use 1).
    """
    batch_size, num_heads, head_dim = query.shape
    _, block_size, num_kv_heads, _ = key_cache.shape
    max_blocks = block_tables.shape[1]
    max_len = max_blocks * block_size
    group = num_heads // num_kv_heads

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    bt = block_tables.long()
    # Gather: [B, max_blocks, block_size, KVH, D] -> [B, KVH, max_len, D]
    k = key_cache[bt].view(batch_size, max_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    v = value_cache[bt].view(batch_size, max_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)

    # GQA: query head h uses kv head h // group  ->  [B, KVH, group, D]
    # Scale first and compute scores in fp32: unscaled fp16 Q.K^T overflows
    # (|q.k| > 65504) on models with large activations such as Qwen2.5-7B.
    q = (query.float() * scale).view(batch_size, num_kv_heads, group, head_dim)

    # Scores: [B, KVH, group, max_len]
    scores = torch.matmul(q, k.float().transpose(-1, -2))
    token_idx = torch.arange(max_len, device=query.device)
    valid = token_idx[None, :] < seq_lens.to(token_idx.dtype)[:, None]   # [B, max_len]
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)

    out = torch.matmul(probs, v)  # [B, KVH, group, D]
    return out.reshape(batch_size, num_heads, head_dim)

