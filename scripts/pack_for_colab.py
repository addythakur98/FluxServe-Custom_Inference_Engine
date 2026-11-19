"""
Zip the engine source so it can be dropped into Google Drive for the Colab server.

    python scripts/pack_for_colab.py
    # -> dist/custom_inference_engine.zip  (upload to MyDrive/ once, and again after code edits)

Weights, caches, the UI, and notebooks are excluded; Colab only needs the Python code.
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "custom_inference_engine.zip"

INCLUDE_DIRS = ["benchmarks", "core", "kernels", "models", "telemetry", "server", "scripts"]
INCLUDE_FILES = ["__init__.py", "config.py", ".env", "README.md"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".safetensors", ".bin", ".pt"}


def iter_files():
    for name in INCLUDE_FILES:
        p = ROOT / name
        if p.is_file():
            yield p
    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and not (set(p.parts) & SKIP_PARTS) and p.suffix not in SKIP_SUFFIXES:
                yield p


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in iter_files():
            zf.write(p, Path("custom_inference_engine") / p.relative_to(ROOT))
            count += 1
    print(f"Packed {count} files -> {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print("Upload it to the root of your Google Drive (MyDrive/custom_inference_engine.zip).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

