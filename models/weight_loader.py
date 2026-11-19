"""
Weight loader for Qwen2.5 (dense) checkpoints.

Loads HuggingFace safetensors weights (single file or sharded) and maps them
onto the custom :class:`Qwen2Model` parameter names::

    model = Qwen2Model(config)
    load_qwen2_weights(model, "./qwen2.5-7b-instruct")
"""

from __future__ import annotations

import json
import os
from typing import Dict, Set

import torch

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Weight Name Mapping
# ---------------------------------------------------------------------------

def _build_weight_name_map(
    num_layers: int = 28,
    tie_word_embeddings: bool = False,
) -> Dict[str, str]:
    """Build a mapping from HuggingFace weight names to our model parameter names.

    HuggingFace uses a ``model.`` prefix (e.g. ``model.embed_tokens.weight``),
    while our :class:`Qwen2Model` uses the bare name (``embed_tokens.weight``).

    When ``tie_word_embeddings`` is False (e.g. Qwen2.5-7B-Instruct),
    ``lm_head.weight`` is mapped to ``lm_head.weight``. When True (e.g. 0.5B),
    ``lm_head.weight`` is omitted and tied to ``embed_tokens.weight``.

    Returns:
        Dict mapping HF name -> our model's parameter name.
    """
    name_map: Dict[str, str] = {}

    # Global parameters
    name_map["model.embed_tokens.weight"] = "embed_tokens.weight"
    name_map["model.norm.weight"] = "norm.weight"
    if not tie_word_embeddings:
        name_map["lm_head.weight"] = "lm_head.weight"

    # Per-layer parameters
    for i in range(num_layers):
        hf_prefix = f"model.layers.{i}"
        our_prefix = f"layers.{i}"

        # Layer norms
        name_map[f"{hf_prefix}.input_layernorm.weight"] = f"{our_prefix}.input_layernorm.weight"
        name_map[f"{hf_prefix}.post_attention_layernorm.weight"] = f"{our_prefix}.post_attention_layernorm.weight"

        # Attention projections
        for proj in ("q_proj", "k_proj", "v_proj"):
            name_map[f"{hf_prefix}.self_attn.{proj}.weight"] = f"{our_prefix}.self_attn.{proj}.weight"
            name_map[f"{hf_prefix}.self_attn.{proj}.bias"] = f"{our_prefix}.self_attn.{proj}.bias"
        # o_proj (no bias)
        name_map[f"{hf_prefix}.self_attn.o_proj.weight"] = f"{our_prefix}.self_attn.o_proj.weight"

        # MLP projections (no bias)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name_map[f"{hf_prefix}.mlp.{proj}.weight"] = f"{our_prefix}.mlp.{proj}.weight"

    return name_map


def _resolve_safetensors_files(model_dir: str) -> list[str]:
    """Return ordered list of safetensors shard paths in ``model_dir``."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        shard_names = sorted(set(index_data.get("weight_map", {}).values()))
        shard_paths = [os.path.join(model_dir, name) for name in shard_names]
        missing_shards = [p for p in shard_paths if not os.path.isfile(p)]
        if missing_shards:
            raise FileNotFoundError(
                f"Missing safetensors shard(s) in {model_dir}: {missing_shards}"
            )
        return shard_paths

    single_path = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single_path):
        return [single_path]

    discovered = sorted(
        os.path.join(model_dir, fn)
        for fn in os.listdir(model_dir)
        if fn.endswith(".safetensors")
    ) if os.path.isdir(model_dir) else []
    if discovered:
        return discovered

    raise FileNotFoundError(
        f"No .safetensors files found in {model_dir}. "
        f"Run snapshot_download or download_model('{model_dir}') first."
    )


# ---------------------------------------------------------------------------
# Weight Loading
# ---------------------------------------------------------------------------

def load_qwen2_weights(
    model: torch.nn.Module,
    model_dir: str,
    dtype: torch.dtype = torch.float16,
    strict: bool = True,
) -> None:
    """Load Qwen2.5 weights (single-file or sharded safetensors) into ``model`` in-place.

    Supports both tied-embedding checkpoints (0.5B / 1.5B / 3B) and untied
    ``lm_head.weight`` checkpoints (7B / 14B / 32B / 72B), as well as multi-shard
    ``model-00001-of-0000N.safetensors`` layouts.
    """
    if safe_open is None:
        raise ImportError(
            "The `safetensors` library is required. "
            "Install it with: pip install safetensors"
        )

    shard_paths = _resolve_safetensors_files(model_dir)

    # Determine number of layers and whether lm_head is tied
    num_layers = len(model.layers) if hasattr(model, "layers") else 28
    has_standalone_lm_head = getattr(model, "lm_head", None) is not None
    tie_word_embeddings = not has_standalone_lm_head
    hf_to_ours = _build_weight_name_map(
        num_layers=num_layers,
        tie_word_embeddings=tie_word_embeddings,
    )

    # Collect the set of our model's parameter names for strict checking
    model_param_names: Set[str] = set(dict(model.named_parameters()).keys())

    loaded_params: Set[str] = set()
    skipped_hf_names: list[str] = []

    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for hf_name in f.keys():
                if hf_name == "lm_head.weight" and tie_word_embeddings:
                    skipped_hf_names.append(hf_name)
                    continue

                our_name = hf_to_ours.get(hf_name)
                if our_name is None:
                    skipped_hf_names.append(hf_name)
                    continue

                tensor = f.get_tensor(hf_name).to(dtype)
                param = _get_nested_attr(model, our_name)

                if param.shape != tensor.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {our_name}: "
                        f"model has {param.shape}, checkpoint has {tensor.shape}"
                    )

                param.data.copy_(tensor)
                loaded_params.add(our_name)

    # Fallback: if model created a standalone lm_head but checkpoint omitted it, tie to embed_tokens
    if has_standalone_lm_head and "lm_head.weight" not in loaded_params and "embed_tokens.weight" in loaded_params:
        model.lm_head.weight = model.embed_tokens.weight
        loaded_params.add("lm_head.weight")

    # Strict mode: verify all model parameters were loaded
    if strict:
        expected_missing: Set[str] = set()
        missing = model_param_names - loaded_params - expected_missing
        if missing:
            raise RuntimeError(
                f"Strict loading failed. {len(missing)} parameter(s) not found "
                f"in checkpoint:\n  " + "\n  ".join(sorted(missing))
            )

    if skipped_hf_names:
        _log(
            f"Skipped {len(skipped_hf_names)} HF tensor(s) "
            f"(tied/unmapped): {skipped_hf_names}"
        )

    _log(
        f"Successfully loaded {len(loaded_params)}/{len(model_param_names)} "
        f"parameters from {len(shard_paths)} safetensors file(s) in {model_dir}"
    )


def _get_nested_attr(obj: object, attr_path: str) -> torch.nn.Parameter:
    """Retrieve a nested attribute by dot-separated path (e.g. 'layers.0.mlp.gate_proj.weight')."""
    parts = attr_path.split(".")
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]  # type: ignore[index]
        else:
            obj = getattr(obj, part)
    return obj  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Simple logging to stdout (avoids external logging dependency)."""
    print(f"[weight_loader] {msg}")
