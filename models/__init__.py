"""Qwen2.5 model definition and HuggingFace safetensors weight loading."""

try:
    from .qwen2_model import Qwen2Config, Qwen2Model
    from .weight_loader import load_qwen2_weights
except ImportError:  # torch optional
    Qwen2Config = None  # type: ignore[assignment,misc]
    Qwen2Model = None  # type: ignore[assignment,misc]
    load_qwen2_weights = None  # type: ignore[assignment]

__all__ = [
    "Qwen2Config",
    "Qwen2Model",
    "load_qwen2_weights",
]
