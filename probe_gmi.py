#!/usr/bin/env python3
"""Probe GMI Cloud APIs — LLM vision (image-to-text) vs video generation queue."""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
TEST_IMAGE = ROOT / "test_images" / "01_empty_hallway.png"

LLM_BASE = os.environ.get("GMI_LLM_ENDPOINT_URL", "https://api.gmi-serving.com").rstrip("/")
VIDEO_BASE = os.environ.get(
    "GMI_VIDEO_ENDPOINT_URL", "https://console.gmicloud.ai"
).rstrip("/")
TIMEOUT_S = 20


def _api_key() -> str:
    key = os.environ.get("GMI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GMI_API_KEY is not set. Export it before running:\n"
            "  export GMI_API_KEY='your-key-here'"
        )
    return key


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _pretty(data: object, limit: int = 6000) -> str:
    try:
        text = json.dumps(data, indent=2, ensure_ascii=False)
    except TypeError:
        text = repr(data)
    return text if len(text) <= limit else text[:limit] + f"\n... [{len(text)} chars total]"


def _get(url: str) -> requests.Response:
    print(f"\n>>> GET {url}")
    resp = requests.get(url, headers=_headers(), timeout=TIMEOUT_S)
    print(f"    status: {resp.status_code}")
    try:
        print(_pretty(resp.json()))
    except ValueError:
        print(resp.text[:3000])
    return resp


def _pick_vision_model(models_body: dict) -> str | None:
    data = models_body.get("data", [])
    preferred_substrings = (
        "mimo-v2.5",
        "mimo",
        "qwen-vl",
        "qwen2-vl",
        "gpt-5",
        "gemini",
        "llava",
        "vision",
    )
    ids = [m.get("id", "") for m in data if isinstance(m, dict)]
    for needle in preferred_substrings:
        for mid in ids:
            if needle in mid.lower():
                return mid
    # fallback: first model
    return ids[0] if ids else None


def main() -> int:
    print("=" * 60)
    print("GMI CLOUD API PROBE")
    print("=" * 60)
    print(f"LLM base:   {LLM_BASE}")
    print(f"Video base: {VIDEO_BASE}")

    try:
        _api_key()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\n--- LLM models (image-to-text / multimodal chat) ---")
    llm_models = _get(f"{LLM_BASE}/v1/models")
    vision_model = os.environ.get("GMI_VISION_MODEL", "").strip()
    if not vision_model and llm_models.status_code == 200:
        try:
            vision_model = _pick_vision_model(llm_models.json()) or ""
        except ValueError:
            vision_model = ""
    if vision_model:
        print(f"\nSelected vision model: {vision_model}")
    else:
        print("\nWARNING: could not pick a vision model from /v1/models", file=sys.stderr)

    print("\n--- Video queue models (image-to-video generation — NOT vision understanding) ---")
    _get(f"{VIDEO_BASE}/api/v1/ie/requestqueue/apikey/models")

    vision_ok = False
    vision_reply = ""
    if vision_model and TEST_IMAGE.is_file():
        print("\n--- Image-to-text probe (chat completions + image_url) ---")
        mime = "image/png" if TEST_IMAGE.suffix.lower() == ".png" else "image/jpeg"
        b64 = base64.b64encode(TEST_IMAGE.read_bytes()).decode("ascii")
        payload = {
            "model": vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this hallway in one short sentence."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            "max_completion_tokens": 64,
            "temperature": 0.1,
        }
        url = f"{LLM_BASE}/v1/chat/completions"
        print(f"\n>>> POST {url}")
        resp = requests.post(url, headers=_headers(), json=payload, timeout=TIMEOUT_S)
        print(f"    status: {resp.status_code}")
        try:
            body = resp.json()
            print(_pretty(body))
            if resp.status_code == 200:
                vision_ok = True
                vision_reply = str(body["choices"][0]["message"]["content"])
        except (ValueError, KeyError, IndexError, TypeError):
            print(resp.text[:3000])
    else:
        print("\nSkipping image probe — no model or test image missing.")

    print("\n" + "=" * 60)
    print("GMI PROBE CONCLUSIONS")
    print("=" * 60)
    print("For evacuation vision (hallway hazard detection), use:")
    print(f"  POST {LLM_BASE}/v1/chat/completions")
    print("  with image_url in messages (image-to-text / multimodal LLM).")
    print("\nThe video request-queue API generates videos FROM images — it does NOT")
    print("describe scenes or return hazard guidance text.")
    print(
        f"\nENDPOINT SUPPORTS: [vision/image-to-text: {'yes' if vision_ok else 'no'}] "
        f"[video-to-text: not confirmed — use still frames for now] "
        f"[video generation: separate queue API] "
        f"[model: {vision_model or 'unknown'}]"
    )
    if vision_ok:
        print(f"Sample reply: {vision_reply!r}")
    return 0 if vision_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
