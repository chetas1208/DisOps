#!/usr/bin/env python3
"""DisOps entry point.

Run the real-time disaster-response web app:

    python app.py                 # serves on http://0.0.0.0:8000
    HOST=127.0.0.1 PORT=9000 python app.py

The frontend (served from ./web) connects over a WebSocket and streams the
Camera → Vision → Cloudflare → Kimi K2 → Voice pipeline live. Demo Mode is on by
default, so it runs end-to-end with no API keys. Provide GMI_API_KEY /
GEMINI_API_KEY / NEMOTRON_ENDPOINT_URL to enable real live-camera analysis.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    reload_flag = os.environ.get("RELOAD", "").lower() in {"1", "true", "yes"}
    uvicorn.run("server.api:app", host=host, port=port, reload=reload_flag)


if __name__ == "__main__":
    main()
