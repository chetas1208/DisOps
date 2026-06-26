"""Nemotron Omni video chunk analysis — /v1/video/analyze."""

from __future__ import annotations

import os
from pathlib import Path

import requests

DEFAULT_BASE_URL = "https://vermont-utah-bidder-rod.trycloudflare.com"
REQUEST_TIMEOUT_S = float(os.environ.get("OMNI_VIDEO_TIMEOUT_S", "120"))

OMNI_VIDEO_PROMPT = (
    "You are monitoring a hallway via a security camera for an evacuation guidance system. "
    "This is a short clip (about 3 seconds). Describe what you see across the clip: any hazards "
    "(crowd, smoke, blockage, debris, fire), how the scene changes frame to frame, and which "
    "direction (left/right/straight) has the clearest path. Be factual and concise (2-4 sentences)."
)


def _base_url() -> str:
    return os.environ.get("NEMOTRON_ENDPOINT_URL", DEFAULT_BASE_URL).strip().rstrip("/")


def _is_stub_response(text: str) -> bool:
    lowered = text.lower()
    return (
        "[mock omni" in lowered
        or "deterministic stub" in lowered
        or '"mock": true' in lowered
        or "stub response" in lowered
    )


def _extract_result_text(body: dict) -> str:
    if body.get("status") != "succeeded":
        raise RuntimeError(f"Omni video job failed: {body.get('error') or body}")

    result = body.get("result") or {}
    text = result.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"Omni video returned empty analysis: {body!r}")
    return text.strip()


def analyze_video_chunk(video_path: str | Path, *, prompt: str | None = None) -> str:
    """
    Send a ~3s video chunk to Nemotron Omni for temporal scene analysis.

    Returns raw analysis text (pass to Kimi reasoning layer).
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    url = f"{_base_url()}/v1/video/analyze"
    wait_timeout = str(int(REQUEST_TIMEOUT_S))

    with path.open("rb") as handle:
        try:
            response = requests.post(
                url,
                files={"file": (path.name, handle, "video/mp4")},
                data={
                    "prompt": prompt or OMNI_VIDEO_PROMPT,
                    "wait": "true",
                    "wait_timeout": wait_timeout,
                    "max_frames": os.environ.get("OMNI_MAX_FRAMES", "8"),
                },
                timeout=REQUEST_TIMEOUT_S + 10,
            )
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Omni video timed out after {REQUEST_TIMEOUT_S}s for {path.name}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Omni video request failed for {path.name}: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Omni video HTTP {response.status_code} for {path.name}: {response.text[:300]}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Omni video returned non-JSON for {path.name}") from exc

    text = _extract_result_text(body)
    if _is_stub_response(text):
        raise RuntimeError(
            f"Omni video returned mock/stub output for {path.name} "
            "(real Omni model not loaded on server)"
        )
    return text
