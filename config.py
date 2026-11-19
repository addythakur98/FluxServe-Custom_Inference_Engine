"""
Centralized Environment & Engine Configuration (``config.py``).

Loads configuration from a ``.env`` file (if present) and environment
variables, providing a single source of truth across:
    - ``weight_loader.py``
    - ``qwen2_model.py``
    - ``engine.py``
    - ``live_monitor.py``
    - ``benchmark_colab.ipynb``

To switch models, hardware (CUDA / CPU), KV-cache capacity, or benchmark
parameters, edit ``.env`` or set ``os.environ["ENGINE_MODEL_REPO"] = ...``
before calling ``get_settings()`` / ``reload_settings()``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def load_dotenv(dotenv_path: Optional[str] = None, override: bool = False) -> None:
    """Minimal zero-dependency ``.env`` loader."""
    candidates = (
        [Path(dotenv_path)]
        if dotenv_path
        else [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]
    )
    for path in candidates:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip("'").strip('"')
                    if key and (override or key not in os.environ):
                        os.environ[key] = val
            break


def _env_str(key: str, default: str) -> str:
    val = os.environ.get(key, "").strip()
    return val if val else default


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key, "").strip()
    return int(val) if val else default


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key, "").strip()
    return float(val) if val else default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


@dataclass
class EngineSettings:
    """Global settings for model, device, KV cache, scheduler, and telemetry."""

    # Model & hardware
    model_repo: str = "Qwen/Qwen2.5-7B-Instruct"
    model_dir: str = ""
    device_preference: str = "auto"   # "auto" | "cuda" | "cpu"
    dtype_preference: str = "auto"    # "auto" | "float16" | "bfloat16" | "float32"

    # Paged KV cache & Continuous Batching scheduler
    num_blocks: int = 512
    block_size: int = 16
    max_batch_size: int = 16
    max_batched_tokens: int = 4096
    max_model_len: int = 2048
    enable_prefix_caching: bool = True
    use_cuda_graph: bool = True
    temperature: float = 0.0

    # Live telemetry & benchmark workload
    trace_num_requests: int = 60
    trace_qps: float = 10.0
    trace_max_new_tokens: int = 48
    trace_burst_initial: int = 8
    trace_stream_bytes: int = 1_500_000
    benchmark_num_requests: int = 100
    benchmark_max_new_tokens: int = 64

    def __post_init__(self) -> None:
        if not self.model_dir:
            slug = self.model_repo.split("/")[-1].lower()
            self.model_dir = f"./{slug}"

    @property
    def model_name(self) -> str:
        return self.model_repo.split("/")[-1]

    def resolve_device(self) -> str:
        if self.device_preference != "auto":
            return self.device_preference
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def resolve_torch_dtype(self):
        import torch
        dev = self.resolve_device()
        pref = self.dtype_preference.lower()
        if pref == "auto":
            return torch.float16 if dev.startswith("cuda") else torch.float32
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if pref not in mapping:
            raise ValueError(f"Unsupported ENGINE_DTYPE={self.dtype_preference!r}")
        return mapping[pref]

    @classmethod
    def from_env(cls, dotenv_path: Optional[str] = None, override: bool = False) -> "EngineSettings":
        load_dotenv(dotenv_path=dotenv_path, override=override)
        return cls(
            model_repo=_env_str("ENGINE_MODEL_REPO", "Qwen/Qwen2.5-7B-Instruct"),
            model_dir=_env_str("ENGINE_MODEL_DIR", ""),
            device_preference=_env_str("ENGINE_DEVICE", "auto"),
            dtype_preference=_env_str("ENGINE_DTYPE", "auto"),
            num_blocks=_env_int("ENGINE_NUM_BLOCKS", 512),
            block_size=_env_int("ENGINE_BLOCK_SIZE", 16),
            max_batch_size=_env_int("ENGINE_MAX_BATCH_SIZE", 16),
            max_batched_tokens=_env_int("ENGINE_MAX_BATCHED_TOKENS", 4096),
            max_model_len=_env_int("ENGINE_MAX_MODEL_LEN", 2048),
            enable_prefix_caching=_env_bool("ENGINE_ENABLE_PREFIX_CACHING", True),
            use_cuda_graph=_env_bool("ENGINE_USE_CUDA_GRAPH", True),
            temperature=_env_float("ENGINE_TEMPERATURE", 0.0),
            trace_num_requests=_env_int("ENGINE_TRACE_NUM_REQUESTS", 60),
            trace_qps=_env_float("ENGINE_TRACE_QPS", 10.0),
            trace_max_new_tokens=_env_int("ENGINE_TRACE_MAX_NEW_TOKENS", 48),
            trace_burst_initial=_env_int("ENGINE_TRACE_BURST_INITIAL", 8),
            trace_stream_bytes=_env_int("ENGINE_TRACE_STREAM_BYTES", 1_500_000),
            benchmark_num_requests=_env_int("ENGINE_BENCHMARK_NUM_REQUESTS", 100),
            benchmark_max_new_tokens=_env_int("ENGINE_BENCHMARK_MAX_NEW_TOKENS", 64),
        )


SETTINGS = EngineSettings.from_env()


def reload_settings(dotenv_path: Optional[str] = None, override: bool = True) -> EngineSettings:
    """Re-read ``.env`` / environment variables and update ``SETTINGS`` in-place."""
    global SETTINGS
    SETTINGS = EngineSettings.from_env(dotenv_path=dotenv_path, override=override)
    return SETTINGS

