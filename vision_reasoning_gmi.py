"""GMI Kimi reasoning — turn Omni video analysis into evacuation decisions."""

from __future__ import annotations

import json
import os
import re

import requests

from vision_core import GuidanceResult

DEFAULT_BASE_URL = "https://api.gmi-serving.com"
DEFAULT_MODEL = "moonshotai/kimi-k2.7-code-highspeed"
REQUEST_TIMEOUT_S = float(os.environ.get("GMI_REQUEST_TIMEOUT_S", "30"))


def _api_key() -> str:
    key = os.environ.get("GMI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GMI_API_KEY is not set.")
    return key


def _reasoning_model() -> str:
    return os.environ.get("GMI_REASONING_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _base_url() -> str:
    return os.environ.get("GMI_LLM_ENDPOINT_URL", DEFAULT_BASE_URL).strip().rstrip("/")


def _format_prior(
    prior_omni_text: str | None,
    prior_decision: GuidanceResult | None,
) -> str:
    if not prior_omni_text and not prior_decision:
        return "none (first chunk)"

    parts: list[str] = []
    if prior_decision:
        parts.append(
            f"last guidance: hazard={prior_decision['hazard']}, "
            f"direction={prior_decision['direction']}, text={prior_decision['text']!r}"
        )
    if prior_omni_text:
        parts.append(f"last camera analysis: {prior_omni_text}")
    return " | ".join(parts)


def _parse_json_decision(raw: str) -> GuidanceResult:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Kimi returned non-JSON decision: {raw[:300]!r}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Kimi JSON must be an object, got: {type(data).__name__}")

    hazard = data.get("hazard")
    direction = data.get("direction")
    spoken = data.get("text")

    if not isinstance(hazard, bool):
        raise RuntimeError(f"Kimi JSON missing bool hazard: {data!r}")
    if not isinstance(spoken, str) or not spoken.strip():
        raise RuntimeError(f"Kimi JSON missing text: {data!r}")
    if hazard and direction not in {"left", "right", "straight", "stop"}:
        raise RuntimeError(f"Kimi JSON invalid direction for hazard=true: {data!r}")

    return {
        "hazard": hazard,
        "direction": direction if direction else None,
        "text": spoken.strip(),
    }


def decide_from_omni(
    omni_text: str,
    *,
    prior_omni_text: str | None = None,
    prior_decision: GuidanceResult | None = None,
) -> GuidanceResult:
    """
    Use GMI Kimi to interpret Omni's video analysis and detect meaningful changes.
    """
    prior = _format_prior(prior_omni_text, prior_decision)
    user_prompt = f"""You are an evacuation guidance system. A security camera sent a 3-second video chunk.

Previous state:
{prior}

Current camera analysis (from Omni video model):
{omni_text}

Compare to the previous state. Decide evacuation guidance for RIGHT NOW.
Return ONLY valid JSON (no markdown):
{{"hazard": true|false, "direction": "left"|"right"|"straight"|"stop"|null, "text": "one short spoken sentence"}}

Rules:
- hazard=false and direction=null when path is clear (text like "clear, continue.")
- hazard=true requires direction left/right/straight/stop
- text must be one concise sentence suitable to speak aloud
- only change guidance if the scene meaningfully changed"""

    payload = {
        "model": _reasoning_model(),
        "messages": [{"role": "user", "content": user_prompt}],
        "max_completion_tokens": int(os.environ.get("GMI_MAX_COMPLETION_TOKENS", "256")),
    }

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
        raise RuntimeError(f"Kimi reasoning timed out after {REQUEST_TIMEOUT_S}s") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Kimi reasoning request failed: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(f"Kimi reasoning HTTP {response.status_code}: {response.text[:300]}")

    body = response.json()
    if body.get("error"):
        raise RuntimeError(f"Kimi reasoning error: {body['error']}")

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Kimi response: {body!r}") from exc

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Kimi returned empty reasoning content")

    return _parse_json_decision(content)
