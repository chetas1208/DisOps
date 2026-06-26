#!/usr/bin/env python3
"""DisOps entry point.

Run the real-time disaster-response web app:

    python app.py                 # serves on http://0.0.0.0:8000
    HOST=127.0.0.1 PORT=9000 python app.py

The frontend (served from ./web) connects over a WebSocket and streams the
Camera → GPT-5.5 Vision → Kimi K2 → Voice pipeline live. Demo Mode is off by
default — start a session to use your camera. Provide GMI_API_KEY for live analysis.
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a local .env into os.environ (no dependency).

    Existing environment variables always win, so explicit exports override the
    file. Missing .env is fine — the app falls back to Demo Mode.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    _load_dotenv()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    reload_flag = os.environ.get("RELOAD", "").lower() in {"1", "true", "yes"}
    uvicorn.run("server.api:app", host=host, port=port, reload=reload_flag)


if __name__ == "__main__":
    main()
