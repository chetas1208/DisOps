"""Nemotron vision backend — same contract as vision_core.analyze_hallway."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

import requests

from vision_core import PROMPT, GuidanceResult, _parse_guidance

DEFAULT_BASE_URL = "https://vermont-utah-bidder-rod.trycloudflare.com"
OMNI_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
REQUEST_TIMEOUT_S = 3.0


def _base_url() -> str:
    return os.environ.get("NEMOTRON_ENDPOINT_URL", DEFAULT_BASE_URL).strip().rstrip("/")


def _is_stub_response(text: str) -> bool:
    lowered = text.lower()
    return (
        "[mock omni" in lowered
        or "deterministic stub" in lowered
        or "stub response" in lowered
    )


def _extract_chat_text(body: dict) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Nemotron chat response shape: {body!r}") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Nemotron returned empty chat content")
    return content.strip()


def analyze_hallway(image_path: str | Path) -> GuidanceResult:
    """
    Analyze a hallway image via Nemotron Omni (/v1/chat/completions).

    Returns the same shape as vision_core.analyze_hallway.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")

    payload = {
        "model": OMNI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 64,
        "temperature": 0.1,
    }

    url = f"{_base_url()}/v1/chat/completions"
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_S)
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Nemotron vision timed out after {REQUEST_TIMEOUT_S}s for {path.name}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Nemotron vision request failed for {path.name}: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Nemotron vision HTTP {response.status_code} for {path.name}: {response.text[:300]}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Nemotron vision returned non-JSON for {path.name}") from exc

    raw_text = _extract_chat_text(body)
    if _is_stub_response(raw_text):
        raise RuntimeError(
            f"Nemotron vision returned mock/stub output for {path.name} "
            "(real Omni model not loaded on server)"
        )

    return _parse_guidance(raw_text)
