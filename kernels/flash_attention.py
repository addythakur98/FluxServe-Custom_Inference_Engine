"""Prefill attention via PyTorch SDPA (dispatches to the FlashAttention-2 kernel on CUDA when eligible)."""

from typing import Optional

import torch
import torch.nn.functional as F


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Prefill attention over [batch, heads, seq, head_dim] tensors.

    Args:
        attn_mask: optional boolean mask broadcastable to
            [batch, heads, seq_len_q, seq_len_k] (True = attend). When given,
            ``causal`` is ignored. Used for prefill on top of prefix-cached
            context, where queries sit at an offset inside the key sequence.
            (An explicit mask makes SDPA use the memory-efficient backend
            instead of the FlashAttention backend.)
    """
    if attn_mask is not None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
