"""
Colab launcher: GPU inference server + public tunnel in one call.

From a Colab cell (after mounting Drive and ``cd``-ing into the repo)::

    from server.colab_launcher import launch
    info = launch(api_key="my-secret")      # prints the public URL
    info["url"]                               # paste into the UI's settings

What it does:
    1. Ensures the HF model is on **Google Drive** (downloads once; every later
       session loads straight from Drive, no re-download).
    2. Starts ``uvicorn server.api_server:app`` as a background process.
    3. Opens a public HTTPS tunnel (Cloudflare quick tunnel by default, no
       account needed; or ngrok with an auth token).
    4. Waits until the engine reports ``ready`` and prints the URL + key.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_DRIVE_MODELS = "/content/drive/MyDrive/llm_models"
LOG_DIR = Path("/content/engine_logs") if Path("/content").is_dir() else REPO_ROOT / "logs"

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"

# Tuned for a 0.5B model on an A100: ~12 KB of KV per token -> 4096 blocks x 16
# tokens = 65k cached tokens (~0.8 GB of VRAM).
DEFAULT_ENGINE_ENV = {
    "ENGINE_DEVICE": "auto",
    "ENGINE_DTYPE": "auto",
    "ENGINE_NUM_BLOCKS": "4096",
    "ENGINE_BLOCK_SIZE": "16",
    "ENGINE_MAX_BATCH_SIZE": "32",
    "ENGINE_MAX_BATCHED_TOKENS": "8192",
    "ENGINE_MAX_MODEL_LEN": "8192",
    "ENGINE_ENABLE_PREFIX_CACHING": "true",
    "ENGINE_USE_CUDA_GRAPH": "true",
    "ENGINE_TEMPERATURE": "0.0",
}

_procs: Dict[str, subprocess.Popen] = {}


def _log(msg: str) -> None:
    print(f"[launcher] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Model on Drive
# ---------------------------------------------------------------------------

def is_model_complete(model_dir: str) -> bool:
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        return False
    if not os.path.isfile(os.path.join(model_dir, "tokenizer.json")):
        return False
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            shards = set(json.load(f).get("weight_map", {}).values())
        return bool(shards) and all(os.path.isfile(os.path.join(model_dir, s)) for s in shards)
    return os.path.isfile(os.path.join(model_dir, "model.safetensors"))


def ensure_model(model_repo: str = DEFAULT_MODEL_REPO, model_dir: Optional[str] = None) -> str:
    """Return a local dir with the model, downloading to Drive only if missing."""
    model_dir = model_dir or os.path.join(DEFAULT_DRIVE_MODELS, model_repo.split("/")[-1].lower())
    if is_model_complete(model_dir):
        _log(f"Model found on Drive, skipping download: {model_dir}")
        return model_dir

    _log(f"Model not found in {model_dir}; downloading {model_repo} once (saved for future sessions)...")
    from huggingface_hub import snapshot_download

    os.makedirs(model_dir, exist_ok=True)
    snapshot_download(
        model_repo,
        local_dir=model_dir,
        allow_patterns=[
            "*.safetensors", "model.safetensors.index.json", "config.json",
            "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.json", "merges.txt",
        ],
    )
    if not is_model_complete(model_dir):
        raise RuntimeError(f"Download finished but {model_dir} is incomplete: {os.listdir(model_dir)}")
    _log(f"Model saved to {model_dir}")
    return model_dir


# ---------------------------------------------------------------------------
# 2. API server
# ---------------------------------------------------------------------------

def _wait_port_ready(port: int, timeout_s: float, proc: subprocess.Popen, log_path: Path) -> dict:
    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited with code {proc.returncode}. Log tail:\n{_tail(log_path)}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                last = json.loads(r.read())
        except urllib.error.HTTPError as e:  # 503 while loading
            try:
                last = json.loads(e.read())
            except Exception:
                last = {}
        except Exception:
            last = {}
        if last.get("status") == "ready":
            return last
        if last.get("status") == "error":
            raise RuntimeError(f"Engine failed to load: {last.get('error')}\n{_tail(log_path)}")
        time.sleep(2)
    raise TimeoutError(f"Server not ready after {timeout_s}s (last={last}). Log tail:\n{_tail(log_path)}")


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "".join(path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-n:])
    except FileNotFoundError:
        return "(no log)"


def start_server(model_dir: str, model_repo: str, api_key: str, port: int = 8000,
                 engine_env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    stop_all(quiet=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "server.log"

    env = os.environ.copy()
    env.update(DEFAULT_ENGINE_ENV)
    env.update(engine_env or {})
    env.update({
        "ENGINE_MODEL_REPO": model_repo,
        "ENGINE_MODEL_DIR": model_dir,
        "ENGINE_API_KEY": api_key,
        "PYTHONUNBUFFERED": "1",
    })
    cmd = [sys.executable, "-m", "uvicorn", "server.api_server:app",
           "--host", "0.0.0.0", "--port", str(port), "--log-level", "info"]
    _log(f"Starting API server on :{port} (log: {log_path})")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                            stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    _procs["server"] = proc
    return proc


# ---------------------------------------------------------------------------
# 3. Tunnel
# ---------------------------------------------------------------------------

def _ensure_cloudflared() -> str:
    existing = shutil.which("cloudflared")
    if existing:
        return existing
    dest = Path("/content/cloudflared") if Path("/content").is_dir() else REPO_ROOT / "cloudflared"
    if not dest.exists():
        _log("Downloading cloudflared...")
        urllib.request.urlretrieve(CLOUDFLARED_URL, dest)
        dest.chmod(0o755)
    return str(dest)


def start_cloudflare_tunnel(port: int, timeout_s: float = 60) -> str:
    binary = _ensure_cloudflared()
    log_path = LOG_DIR / "tunnel.log"
    proc = subprocess.Popen(
        [binary, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True,
    )
    _procs["tunnel"] = proc
    pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"cloudflared exited. Log:\n{_tail(log_path)}")
        m = pattern.search(log_path.read_text(errors="replace"))
        if m:
            return m.group(0)
        time.sleep(1)
    raise TimeoutError(f"No tunnel URL after {timeout_s}s. Log:\n{_tail(log_path)}")


def start_ngrok_tunnel(port: int, authtoken: str) -> str:
    from pyngrok import conf, ngrok  # pip install pyngrok

    conf.get_default().auth_token = authtoken
    return ngrok.connect(port, "http").public_url


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def launch(
    api_key: Optional[str] = None,
    model_repo: str = DEFAULT_MODEL_REPO,
    model_dir: Optional[str] = None,
    port: int = 8000,
    tunnel: str = "cloudflare",           # "cloudflare" | "ngrok" | "none"
    ngrok_token: Optional[str] = None,
    engine_env: Optional[Dict[str, str]] = None,
    ready_timeout_s: float = 600,
) -> Dict[str, str]:
    """Ensure weights on Drive, start the server + tunnel, return connection info."""
    api_key = api_key or secrets.token_urlsafe(24)
    model_dir = ensure_model(model_repo, model_dir)

    proc = start_server(model_dir, model_repo, api_key, port, engine_env)

    url = f"http://127.0.0.1:{port}"
    if tunnel == "cloudflare":
        url = start_cloudflare_tunnel(port)
    elif tunnel == "ngrok":
        if not ngrok_token:
            raise ValueError("tunnel='ngrok' requires ngrok_token")
        url = start_ngrok_tunnel(port, ngrok_token)

    _log("Waiting for the engine to load weights + capture CUDA graphs...")
    health = _wait_port_ready(port, ready_timeout_s, proc, LOG_DIR / "server.log")

    bar = "=" * 72
    print(f"\n{bar}\n  ENGINE READY  |  {health.get('model')} on {health.get('gpu')}\n"
          f"  API URL : {url}\n  API KEY : {api_key}\n"
          f"  Paste both into the UI's settings. Keep this Colab tab open.\n{bar}\n", flush=True)
    return {"url": url, "api_key": api_key, "model_dir": model_dir}


def stop_all(quiet: bool = False) -> None:
    for name, proc in list(_procs.items()):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            if not quiet:
                _log(f"Stopped {name}")
        _procs.pop(name, None)


def server_logs(n: int = 60) -> None:
    print(_tail(LOG_DIR / "server.log", n))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Start the inference API server + public tunnel")
    ap.add_argument("--api-key")
    ap.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    ap.add_argument("--model-dir")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tunnel", choices=["cloudflare", "ngrok", "none"], default="cloudflare")
    ap.add_argument("--ngrok-token")
    a = ap.parse_args()
    launch(a.api_key, a.model_repo, a.model_dir, a.port, a.tunnel, a.ngrok_token)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        stop_all()

