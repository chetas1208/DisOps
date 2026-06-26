#!/usr/bin/env python3
"""Discover Nemotron FastAPI backend: health, models, OpenAPI, vision/audio support."""

from __future__ import annotations

import base64
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE = "https://vermont-utah-bidder-rod.trycloudflare.com"
TIMEOUT_S = 15
ROOT = Path(__file__).resolve().parent
TEST_IMAGE = ROOT / "test_images" / "01_empty_hallway.png"


def _base_url() -> str:
    return os.environ.get("NEMOTRON_ENDPOINT_URL", DEFAULT_BASE).strip().rstrip("/")


def _pretty(data: Any, limit: int = 8000) -> str:
    try:
        text = json.dumps(data, indent=2, ensure_ascii=False)
    except TypeError:
        text = repr(data)
    if len(text) > limit:
        return text[:limit] + f"\n... [truncated, {len(text)} chars total]"
    return text


def _get(path: str) -> tuple[int, dict[str, str], Any]:
    url = f"{_base_url()}{path}"
    print(f"\n>>> GET {url}")
    resp = requests.get(url, timeout=TIMEOUT_S)
    print(f"    status: {resp.status_code}")
    body: Any
    if "application/json" in resp.headers.get("content-type", ""):
        try:
            body = resp.json()
            print(_pretty(body))
        except ValueError:
            body = resp.text
            print(body[:4000])
    else:
        body = resp.text
        preview = body[:2000] if isinstance(body, str) else body
        print(f"    content-type: {resp.headers.get('content-type')}")
        print(preview)
    return resp.status_code, dict(resp.headers), body


def _post_json(path: str, payload: dict[str, Any]) -> tuple[int, Any]:
    url = f"{_base_url()}{path}"
    print(f"\n>>> POST {url}")
    print(f"    payload: {_pretty(payload, limit=1200)}")
    resp = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=TIMEOUT_S,
    )
    print(f"    status: {resp.status_code}")
    ctype = resp.headers.get("content-type", "")
    if "application/json" in ctype:
        try:
            body = resp.json()
            print(_pretty(body))
            return resp.status_code, body
        except ValueError:
            pass
    body = resp.text
    print(body[:4000] if isinstance(body, str) else body)
    return resp.status_code, body


def _analyze_openapi(spec: dict[str, Any]) -> dict[str, Any]:
    paths = spec.get("paths", {})
    path_list = sorted(paths.keys())

    chat_paths = [p for p in path_list if "chat" in p.lower() or "completion" in p.lower()]
    audio_paths = [
        p for p in path_list
        if any(k in p.lower() for k in ("audio", "speech", "tts", "voice"))
    ]
    vision_hints: list[str] = []

    # Inspect chat completion request schemas for image support
    for path, methods in paths.items():
        if "chat" not in path.lower() and "completion" not in path.lower():
            continue
        for method, detail in methods.items():
            schema_text = json.dumps(detail).lower()
            if "image_url" in schema_text or "image/png" in schema_text:
                vision_hints.append(f"{method.upper()} {path} schema mentions image_url")
            if "input_audio" in schema_text or "audio_url" in schema_text:
                vision_hints.append(f"{method.upper()} {path} schema mentions audio input")

    return {
        "all_paths": path_list,
        "chat_paths": chat_paths,
        "audio_paths": audio_paths,
        "vision_hints": vision_hints,
    }


def _probe_chat_vision(model: str) -> tuple[bool, str]:
    b64 = base64.b64encode(TEST_IMAGE.read_bytes()).decode("ascii") if TEST_IMAGE.is_file() else None
    if not b64:
        return False, "test image missing"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one short sentence."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 64,
        "temperature": 0,
    }
    status, body = _post_json("/v1/chat/completions", payload)
    if status == 200 and isinstance(body, dict):
        try:
            text = body["choices"][0]["message"]["content"]
            return True, str(text)
        except (KeyError, IndexError, TypeError):
            return True, str(body)
    return False, str(body)[:500]


def _probe_chat_text(model: str) -> tuple[bool, str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: probe-ok"}],
        "max_tokens": 16,
        "temperature": 0,
    }
    status, body = _post_json("/v1/chat/completions", payload)
    if status == 200 and isinstance(body, dict):
        try:
            return True, str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            return True, str(body)
    return False, str(body)[:500]


def _probe_voice_respond() -> tuple[bool, str]:
    url = f"{_base_url()}/v1/voice/respond"
    print(f"\n>>> POST {url} (multipart text)")
    try:
        resp = requests.post(
            url,
            data={"text": "Clear ahead, continue.", "wait": "true"},
            timeout=TIMEOUT_S,
        )
        print(f"    status: {resp.status_code}")
        if resp.status_code != 200:
            print(resp.text[:2000])
            return False, f"HTTP {resp.status_code}"
        body = resp.json()
        print(_pretty(body, limit=3000))
        result = body.get("result") or {}
        audio_url = result.get("audio_url")
        if body.get("status") == "succeeded" and audio_url:
            audio_resp = requests.get(f"{_base_url()}{audio_url}", timeout=TIMEOUT_S)
            ctype = audio_resp.headers.get("content-type", "")
            print(f"    audio fetch: {audio_resp.status_code} {ctype} {len(audio_resp.content)} bytes")
            if audio_resp.status_code == 200 and audio_resp.content:
                return True, f"/v1/voice/respond -> {audio_url} ({ctype})"
        return False, "voice/respond did not return fetchable audio"
    except requests.RequestException as exc:
        print(f"    ERROR: {exc}")
        return False, str(exc)


def main() -> int:
    base = _base_url()
    print("=" * 60)
    print("NEMOTRON ENDPOINT PROBE")
    print("=" * 60)
    print(f"Base URL: {base}")

    # 1. Health
    print("\n" + "=" * 60)
    print("1. GET /health")
    print("=" * 60)
    health_status, _, health_body = _get("/health")
    service_up = health_status == 200

    # Root info
    print("\n" + "=" * 60)
    print("1b. GET / (service info)")
    print("=" * 60)
    _get("/")

    # 2. Models
    print("\n" + "=" * 60)
    print("2. GET /v1/models")
    print("=" * 60)
    models_status, _, models_body = _get("/v1/models")

    model_ids: list[str] = []
    if isinstance(models_body, dict):
        for item in models_body.get("data", models_body.get("models", [])):
            if isinstance(item, dict):
                mid = item.get("id") or item.get("name")
                if mid:
                    model_ids.append(str(mid))
            elif isinstance(item, str):
                model_ids.append(item)
    if not model_ids and isinstance(models_body, list):
        model_ids = [str(m) for m in models_body]

    primary_model = model_ids[0] if model_ids else "nemotron"

    # 3. Docs + OpenAPI
    print("\n" + "=" * 60)
    print("3. GET /docs")
    print("=" * 60)
    _get("/docs")

    print("\n" + "=" * 60)
    print("3b. GET /openapi.json")
    print("=" * 60)
    openapi_status, _, openapi_body = _get("/openapi.json")

    openapi_analysis: dict[str, Any] = {}
    if isinstance(openapi_body, dict):
        openapi_analysis = _analyze_openapi(openapi_body)
        print("\n--- OpenAPI path summary ---")
        print("All paths:")
        for p in openapi_analysis["all_paths"]:
            print(f"  {p}")
        print("\nChat/completion paths:")
        for p in openapi_analysis["chat_paths"]:
            print(f"  {p}")
        print("\nAudio/speech paths:")
        for p in openapi_analysis["audio_paths"] or ["(none found)"]:
            print(f"  {p}")
        if openapi_analysis["vision_hints"]:
            print("\nVision schema hints:")
            for hint in openapi_analysis["vision_hints"]:
                print(f"  {hint}")

    # 4. Live probes
    print("\n" + "=" * 60)
    print("4. Live chat completions probe (text)")
    print("=" * 60)
    text_ok, text_reply = _probe_chat_text(primary_model)
    print(f"Text chat result: ok={text_ok}  reply={text_reply!r}")

    print("\n" + "=" * 60)
    print("4b. Live chat completions probe (vision / image)")
    print("=" * 60)
    vision_ok, vision_reply = _probe_chat_vision(primary_model)
    print(f"Vision chat result: ok={vision_ok}  reply={vision_reply!r}")

    print("\n" + "=" * 60)
    print("4c. Audio/TTS endpoint probe")
    print("=" * 60)
    audio_ok, audio_detail = _probe_voice_respond()
    print(f"Audio probe result: ok={audio_ok}  detail={audio_detail!r}")

    schema_vision = bool(openapi_analysis.get("vision_hints"))
    vision_supported = vision_ok
    audio_supported = audio_ok

    print("\n" + "=" * 60)
    print("PROBE CONCLUSIONS")
    print("=" * 60)
    print(f"Service up (/health):     {'yes' if service_up else 'no'} ({health_status})")
    print(f"Models endpoint (/v1/models): {'ok' if models_status == 200 else f'failed ({models_status})'}")
    print(f"OpenAPI available:        {'yes' if openapi_status == 200 else 'no'}")
    print(f"Text chat works:          {'yes' if text_ok else 'no'}")
    print(f"Vision image input works: {'yes' if vision_ok else 'no'}")
    print(f"Audio/TTS works:          {'yes' if audio_ok else 'no'}")

    print(
        f"\nENDPOINT SUPPORTS: [vision: {'yes' if vision_supported else 'no'}] "
        f"[audio/TTS: {'yes' if audio_supported else 'no'}] "
        f"[model names found: {', '.join(model_ids) if model_ids else 'none'}]"
    )

    if textwrap:
        print(
            textwrap.dedent(
                """
                Interpretation:
                  - vision=yes means /v1/chat/completions accepted image_url content and returned 200.
                  - audio/TTS=yes means an audio/speech endpoint returned audio bytes.
                  - Use model name(s) above in integration code.
                """
            ).strip()
        )

    return 0 if service_up else 1


if __name__ == "__main__":
    raise SystemExit(main())
