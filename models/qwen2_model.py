"""
Qwen2.5 (dense) model definition for the custom inference engine.

Implements the Qwen2.5 transformer in PyTorch, matching the HuggingFace
config.json of any dense Qwen2/Qwen2.5 checkpoint (0.5B ... 7B+):
    - pre-norm (RMSNorm) decoder layers
    - Grouped-Query Attention with QKV bias
    - SwiGLU MLP
    - RoPE (theta from config)
    - tied (small models) or standalone (7B+) lm_head

Two forward paths are provided:
    1. forward_prefill  – full-sequence attention via FlashAttention / SDPA.
       When paged KV caches are supplied, the prompt's K/V are written into
       the paged cache, and (with prefix caching) the new tokens attend over
       previously cached context blocks.
    2. forward_decode   – single-token generation via PagedAttention
       (static-shape, CUDA-Graph capturable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..kernels.paged_attention import paged_attention_decode
    from ..kernels.flash_attention import flash_attention_prefill
except ImportError:
    from kernels.paged_attention import paged_attention_decode
    from kernels.flash_attention import flash_attention_prefill


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

@dataclass
class Qwen2Config:
    """Qwen2.5 configuration (defaults: Qwen2.5-0.5B-Instruct)."""

    vocab_size: int = 151936
    hidden_size: int = 896
    num_attention_heads: int = 14       # query heads
    num_key_value_heads: int = 2        # GQA KV heads
    head_dim: int = 64                  # hidden_size // num_attention_heads
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    dtype: torch.dtype = field(default=torch.float16)
    # Length of the pre-computed RoPE table (keeps memory bounded; the
    # full 32k table is only needed for very long contexts).
    max_model_len: int = 4096

    @classmethod
    def from_hf_config(cls, raw: dict, **overrides) -> "Qwen2Config":
        """Build from a HuggingFace ``config.json`` dictionary."""
        num_heads = raw["num_attention_heads"]
        cfg = dict(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            num_attention_heads=num_heads,
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim", raw["hidden_size"] // num_heads),
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            max_position_embeddings=raw["max_position_embeddings"],
            rope_theta=raw.get("rope_theta", 1_000_000.0),
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            tie_word_embeddings=raw.get("tie_word_embeddings", True),
        )
        cfg.update(overrides)
        return cls(**cfg)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with configurable base frequency (theta).

    Pre-computes the cos/sin tables up to ``max_seq_len`` and caches them as
    non-persistent buffers.  The ``forward`` method accepts an explicit
    ``positions`` tensor so it works for both prefill (arbitrary range) and
    decode (single position per sequence). A single instance is shared by
    all layers.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 4096,
        theta: float = 1_000_000.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Inverse frequency vector: [head_dim // 2]
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # [seq_len, head_dim // 2]
        freqs = torch.outer(positions, inv_freq)
        # [seq_len, head_dim]  (duplicate for paired dims)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _apply(self, fn, recurse: bool = True):
        """Keep the RoPE tables in fp32 even when the model is cast to fp16/bf16.

        RoPE phases grow with position; storing cos/sin in fp16 loses angular
        precision on long contexts, so we only cast at gather time (matching
        HuggingFace's eager rotary embedding).
        """
        super()._apply(fn, recurse)
        self.cos_cached = self.cos_cached.float()
        self.sin_cached = self.sin_cached.float()
        return self

    def forward(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) gathered at ``positions`` ([batch, seq_len])."""
        cos = self.cos_cached[positions].to(dtype)  # [batch, seq_len, head_dim]
        sin = self.sin_cached[positions].to(dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by splitting in half and negating."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        cos: [batch, seq_len, head_dim]
        sin: [batch, seq_len, head_dim]

    Returns:
        Rotated (q, k) with the same shapes.
    """
    # Unsqueeze for head dimension broadcast: [batch, 1, seq_len, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def write_kv_to_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter new K/V vectors into the paged cache (in-place, graph-safe).

    Args:
        k, v: [num_tokens, num_kv_heads, head_dim]
        key_cache, value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        slot_mapping: [num_tokens] flat slot index = block_id * block_size + offset
    """
    num_kv_heads, head_dim = key_cache.shape[-2], key_cache.shape[-1]
    slots = slot_mapping.long()
    key_cache.view(-1, num_kv_heads, head_dim).index_copy_(0, slots, k.to(key_cache.dtype))
    value_cache.view(-1, num_kv_heads, head_dim).index_copy_(0, slots, v.to(value_cache.dtype))


# ---------------------------------------------------------------------------
# Attention (Grouped-Query Attention)
# ---------------------------------------------------------------------------

class Qwen2Attention(nn.Module):
    """Multi-head attention with Grouped-Query Attention (GQA).

    Qwen2.5 uses 14 query heads and 2 KV heads with head_dim=64.
    Q, K, V projections have bias; O projection does NOT.
    """

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 7
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections – Q/K/V have bias, O does not
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    # ----- helpers -----------------------------------------------------------

    def _project_qkv(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to Q, K, V and reshape for multi-head layout.

        Returns:
            q: [batch, num_heads, seq_len, head_dim]
            k: [batch, num_kv_heads, seq_len, head_dim]
            v: [batch, num_kv_heads, seq_len, head_dim]
        """
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    @staticmethod
    def _expand_kv_heads(
        kv: torch.Tensor, num_kv_groups: int
    ) -> torch.Tensor:
        """Repeat KV heads to match the number of query heads for standard attention.

        Args:
            kv: [batch, num_kv_heads, seq_len, head_dim]
            num_kv_groups: repetition factor (num_heads // num_kv_heads)

        Returns:
            [batch, num_heads, seq_len, head_dim]
        """
        if num_kv_groups == 1:
            return kv
        bsz, num_kv_heads, seq_len, head_dim = kv.shape
        kv = kv[:, :, None, :, :].expand(bsz, num_kv_heads, num_kv_groups, seq_len, head_dim)
        return kv.reshape(bsz, num_kv_heads * num_kv_groups, seq_len, head_dim)

    # ----- prefill path ------------------------------------------------------

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        context_len: int = 0,
        cached_blocks: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-sequence attention for prompt prefill.

        Uses ``flash_attention_prefill`` (PyTorch SDPA / FlashAttention-2 backend).
        """
        bsz, seq_len, _ = hidden_states.shape
        q, k, v = self._project_qkv(hidden_states)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Populate the paged KV cache with the (post-RoPE) keys / values
        if key_cache is not None:
            k_tok = k.transpose(1, 2).reshape(bsz * seq_len, self.num_kv_heads, self.head_dim)
            v_tok = v.transpose(1, 2).reshape(bsz * seq_len, self.num_kv_heads, self.head_dim)
            write_kv_to_cache(k_tok, v_tok, key_cache, value_cache, slot_mapping)

        causal = True
        if context_len > 0:
            # Prefix-cache hit: attend over cached context blocks + newly computed suffix tokens
            assert bsz == 1 and key_cache is not None
            if cached_blocks is None:
                assert block_table is not None
                block_size = key_cache.shape[1]
                cached_blocks = block_table[0, : context_len // block_size].long()
            k_prefix = (
                key_cache[cached_blocks]
                .view(1, context_len, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
                .to(q.dtype)
            )
            v_prefix = (
                value_cache[cached_blocks]
                .view(1, context_len, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
                .to(q.dtype)
            )
            k = torch.cat([k_prefix, k], dim=2)
            v = torch.cat([v_prefix, v], dim=2)
            if seq_len == 1:
                # Single suffix query attends to all keys (no mask needed)
                attn_mask = None
                causal = False
            elif attn_mask is None:
                total_len = context_len + seq_len
                q_pos = torch.arange(seq_len, device=q.device)[:, None] + context_len
                k_pos = torch.arange(total_len, device=q.device)[None, :]
                attn_mask = k_pos <= q_pos

        # Expand KV heads for standard attention
        k = self._expand_kv_heads(k, self.num_kv_groups)
        v = self._expand_kv_heads(v, self.num_kv_groups)

        # FlashAttention / SDPA: expects [batch, heads, seq_len, head_dim]
        attn_out = flash_attention_prefill(q, k, v, causal=causal, attn_mask=attn_mask)

        # Merge heads: [batch, seq_len, num_heads * head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)

    # ----- decode path -------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Single-token decode using PagedAttention.

        Args:
            hidden_states: [batch, 1, hidden_size]
            cos, sin: RoPE tables gathered at the new token positions
            key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            block_tables: [batch, max_blocks_per_seq]
            seq_lens: [batch] current total sequence lengths (including this token)
            slot_mapping: [batch] flat slot indices for writing the new KV

        Returns:
            [batch, 1, hidden_size]
        """
        bsz = hidden_states.size(0)
        q, k, v = self._project_qkv(hidden_states)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Write new KV into paged cache at the designated slots
        # k, v: [batch, num_kv_heads, 1, head_dim] -> [batch, num_kv_heads, head_dim]
        write_kv_to_cache(k.squeeze(2), v.squeeze(2), key_cache, value_cache, slot_mapping)

        # PagedAttention decode (vectorised, no host sync -> CUDA-Graph safe)
        attn_out = paged_attention_decode(
            query=q.squeeze(2),          # [batch, num_heads, head_dim]
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            scale=self.scale,
        )
        # attn_out: [batch, num_heads, head_dim]
        attn_out = attn_out.reshape(bsz, 1, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class Qwen2MLP(nn.Module):
    """SwiGLU MLP: out = down_proj(silu(gate_proj(x)) * up_proj(x)).

    All projections are bias-free in Qwen2.5.
    """

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer Decoder Layer
# ---------------------------------------------------------------------------

class Qwen2DecoderLayer(nn.Module):
    """Single Qwen2.5 transformer block: pre-norm attention + pre-norm MLP."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen2MLP(config)

    def _mlp_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        **cache_kwargs,
    ) -> torch.Tensor:
        """Prefill forward with residual connections."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_prefill(hidden_states, cos, sin, **cache_kwargs)
        hidden_states = residual + hidden_states
        return self._mlp_block(hidden_states)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Decode forward (single-token) with residual connections."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_decode(
            hidden_states, cos, sin, key_cache, value_cache,
            block_tables, seq_lens, slot_mapping,
        )
        hidden_states = residual + hidden_states
        return self._mlp_block(hidden_states)


# ---------------------------------------------------------------------------
# Full Qwen2.5 Model
# ---------------------------------------------------------------------------

class Qwen2Model(nn.Module):
    """Complete Qwen2.5 language model.

    Provides two forward paths:
        - ``forward_prefill``: Process a full prompt via FlashAttention/SDPA.
        - ``forward_decode``:  Generate one token at a time via PagedAttention.

    Weight tying: ``lm_head`` is **not** instantiated as a separate
    ``nn.Linear``; instead, the embedding weight is reused directly in the
    output projection (matching ``tie_word_embeddings: true``).

    Typical parameter naming (matches weight_loader expectations)::

        embed_tokens.weight
        layers.{i}.input_layernorm.weight
        layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}
        layers.{i}.post_attention_layernorm.weight
        layers.{i}.mlp.{gate,up,down}_proj.weight
        norm.weight
    """

    def __init__(self, config: Optional[Qwen2Config] = None) -> None:
        super().__init__()
        if config is None:
            config = Qwen2Config()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output projection: tied (0.5B/1.5B/3B) or standalone lm_head (7B+)
        if config.tie_word_embeddings:
            self.lm_head: Optional[nn.Linear] = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Shared RoPE table
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=min(config.max_model_len, config.max_position_embeddings),
            theta=config.rope_theta,
        )

    def _lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        # Tied lm_head: reuse embedding weight
        return F.linear(hidden_states, self.embed_tokens.weight)

    def forward_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        key_caches: Optional[List[torch.Tensor]] = None,
        value_caches: Optional[List[torch.Tensor]] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        context_len: int = 0,
        last_token_only: bool = False,
    ) -> torch.Tensor:
        """Full-sequence prefill forward pass.

        Args:
            input_ids: [batch, seq_len] token ids.
            positions: [batch, seq_len] position ids.
            key_caches / value_caches: optional per-layer paged caches. When
                provided, the prompt K/V are stored at ``slot_mapping``.
            slot_mapping: [batch * seq_len] flat cache slots for the tokens.
            block_table: [1, num_blocks] (needed when ``context_len > 0``).
            context_len: tokens already in the cache from a prefix-cache hit;
                ``input_ids`` then holds only the uncached suffix.
            last_token_only: return logits of the final position only
                (saves a [seq_len, vocab] projection during serving).

        Returns:
            logits: [batch, seq_len, vocab_size] (or [batch, 1, vocab_size])
        """
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions, hidden_states.dtype)

        # Precompute prefix-cache block indices and causal mask once for all layers
        cached_blocks = None
        attn_mask = None
        if context_len > 0 and key_caches is not None and block_table is not None:
            seq_len = input_ids.size(1)
            block_size = key_caches[0].shape[1]
            cached_blocks = block_table[0, : context_len // block_size].long()
            if seq_len > 1:
                total_len = context_len + seq_len
                q_pos = torch.arange(seq_len, device=input_ids.device)[:, None] + context_len
                k_pos = torch.arange(total_len, device=input_ids.device)[None, :]
                attn_mask = k_pos <= q_pos

        for i, layer in enumerate(self.layers):
            if key_caches is not None:
                hidden_states = layer.forward_prefill(
                    hidden_states, cos, sin,
                    key_cache=key_caches[i], value_cache=value_caches[i],
                    slot_mapping=slot_mapping, block_table=block_table,
                    context_len=context_len,
                    cached_blocks=cached_blocks,
                    attn_mask=attn_mask,
                )
            else:
                hidden_states = layer.forward_prefill(hidden_states, cos, sin)

        if last_token_only:
            hidden_states = hidden_states[:, -1:, :]
        hidden_states = self.norm(hidden_states)
        return self._lm_head(hidden_states)

    def forward_decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        key_caches: List[torch.Tensor],
        value_caches: List[torch.Tensor],
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Single-token decode forward pass using paged KV caches.

        Args:
            input_ids: [batch, 1] token ids for the current step.
            positions: [batch, 1] position ids for the current step.
            key_caches: List of per-layer key cache tensors,
                each [num_blocks, block_size, num_kv_heads, head_dim].
            value_caches: List of per-layer value cache tensors,
                each [num_blocks, block_size, num_kv_heads, head_dim].
            block_tables: [batch, max_blocks_per_seq] block id mapping.
            seq_lens: [batch] current sequence lengths.
            slot_mapping: [batch] flat slot indices for this decode step.

        Returns:
            logits: [batch, 1, vocab_size]
        """
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions, hidden_states.dtype)

        for i, layer in enumerate(self.layers):
            hidden_states = layer.forward_decode(
                hidden_states, cos, sin,
                key_caches[i], value_caches[i],
                block_tables, seq_lens, slot_mapping,
            )

        hidden_states = self.norm(hidden_states)
        return self._lm_head(hidden_states)

    # ----- Convenience -------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        """Total number of parameters (excluding tied duplicates)."""
        return sum(p.numel() for p in self.parameters())

    def allocate_kv_caches(
        self,
        num_blocks: int,
        block_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Allocate paged KV cache tensors for all layers.

        Args:
            num_blocks: Total number of physical blocks in the cache.
            block_size: Number of tokens per block.
            dtype: Data type (defaults to the model's parameter dtype).
            device: Target device.

        Returns:
            (key_caches, value_caches) – lists of length num_hidden_layers,
            each tensor shaped [num_blocks, block_size, num_kv_heads, head_dim].
        """
        if dtype is None:
            dtype = self.embed_tokens.weight.dtype
        if device is None:
            device = self.embed_tokens.weight.device

        shape = (
            num_blocks,
            block_size,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        key_caches = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(self.config.num_hidden_layers)]
        value_caches = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(self.config.num_hidden_layers)]
        return key_caches, value_caches
