"""GMI Cloud vision backend — image-to-text via OpenAI-compatible chat completions."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

import requests

from vision_core import PROMPT, GuidanceResult, _parse_guidance

DEFAULT_BASE_URL = "https://api.gmi-serving.com"
DEFAULT_MODEL = "openai/gpt-5.5"
REQUEST_TIMEOUT_S = float(os.environ.get("GMI_REQUEST_TIMEOUT_S", "30"))


def _api_key() -> str:
    key = os.environ.get("GMI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GMI_API_KEY is not set. Export it before running:\n"
            "  export GMI_API_KEY='your-key-here'"
        )
    return key


def _base_url() -> str:
    return os.environ.get("GMI_LLM_ENDPOINT_URL", DEFAULT_BASE_URL).strip().rstrip("/")


def _vision_model() -> str:
    return os.environ.get("GMI_VISION_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _extract_chat_text(body: dict) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected GMI chat response shape: {body!r}") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("GMI returned empty chat content")
    return content.strip()


def analyze_hallway(image_path: str | Path) -> GuidanceResult:
    """
    Analyze a hallway image via GMI Cloud multimodal chat (/v1/chat/completions).

    Returns the same shape as vision_core.analyze_hallway.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")

    payload: dict = {
        "model": _vision_model(),
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
        "max_completion_tokens": int(os.environ.get("GMI_MAX_COMPLETION_TOKENS", "256")),
    }
    temp = os.environ.get("GMI_TEMPERATURE", "").strip()
    if temp:
        payload["temperature"] = float(temp)

    url = f"{_base_url()}/v1/chat/completions"
    try:
        response = requests.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
            },
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.Timeout as exc:
        raise RuntimeError(
            f"GMI vision timed out after {REQUEST_TIMEOUT_S}s for {path.name}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"GMI vision request failed for {path.name}: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"GMI vision HTTP {response.status_code} for {path.name}: {response.text[:300]}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(f"GMI vision returned non-JSON for {path.name}") from exc

    if body.get("error"):
        raise RuntimeError(f"GMI vision error for {path.name}: {body['error']}")

    raw_text = _extract_chat_text(body)
    return _parse_guidance(raw_text)
