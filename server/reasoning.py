"""GMI Kimi K2 emergency-reasoning step.

Takes the compact scene analysis produced by the vision backend and asks the
GMI Cloud reasoning model (``GMI_REASONING_MODEL``, e.g. Kimi K2) for structured,
user-safe emergency guidance.

Only brief, user-safe output is requested — never long chain-of-thought. The
model is constrained to a small JSON object that maps 1:1 onto the fields the UI
renders (urgency, situation, voice_guidance, reasoning, uncertainty, …).

Everything degrades gracefully: if the key/endpoint is missing or the call
fails, callers fall back to the local heuristic mapping so the session never
breaks.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

DEFAULT_BASE_URL = "https://api.gmi-serving.com"
DEFAULT_MODEL = "moonshotai/kimi-k2.7-code-highspeed"
_VALID_URGENCY = {"Low", "Medium", "High", "Critical"}

_SYSTEM_PROMPT = (
    "You are RescueLens, an emergency-response reasoning agent. You receive a "
    "short description of a live camera scene and any detected hazards. Decide "
    "the safest next action for a person who may be in danger. "
    "Respond ONLY with a compact JSON object — no markdown, no extra text — "
    "with exactly these keys:\n"
    '  "urgency": one of "Low","Medium","High","Critical"\n'
    '  "situation": one short sentence naming the situation\n'
    '  "safest_next_action": one short, direct instruction\n'
    '  "voice_guidance": one calm spoken instruction (<= 20 words) for TTS\n'
    '  "reasoning_summary": one brief, user-safe sentence (NO step-by-step '
    "chain-of-thought)\n"
    '  "uncertainty": one short sentence on what is not confirmed\n'
    '  "next_check": one short sentence on what to verify next\n'
    "Keep every value short and calm. Never expose internal reasoning steps."
)


def _api_key() -> str:
    key = os.environ.get("GMI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GMI_API_KEY is not set")
    return key


def _base_url() -> str:
    return os.environ.get("GMI_LLM_ENDPOINT_URL", DEFAULT_BASE_URL).strip().rstrip("/")


def _model() -> str:
    return os.environ.get("GMI_REASONING_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def reasoning_available() -> bool:
    return bool(os.environ.get("GMI_API_KEY", "").strip())


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of the model response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def reason(
    *,
    scene_summary: str,
    hazards: list[str],
    direction: str | None,
    people_detected: int,
) -> dict[str, Any]:
    """Call GMI Kimi K2 for structured emergency guidance.

    Returns a dict with the keys described in ``_SYSTEM_PROMPT`` plus an
    ``elapsed_ms`` timing field. Raises on any failure so the caller can fall
    back to its heuristic mapping.
    """
    user_payload = {
        "scene_summary": scene_summary,
        "detected_hazards": hazards,
        "suggested_direction": direction,
        "people_detected": people_detected,
    }

    payload = {
        "model": _model(),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        # Reasoning models spend completion budget on internal reasoning_content,
        # so give the visible JSON answer comfortable headroom.
        "max_completion_tokens": max(512, int(os.environ.get("GMI_MAX_COMPLETION_TOKENS", "256"))),
    }
    # Some GMI models (e.g. Kimi K2 highspeed) only accept temperature=1; omit it
    # entirely unless explicitly configured to a supported value.
    temp = os.environ.get("GMI_REASONING_TEMPERATURE", "").strip()
    if temp:
        payload["temperature"] = float(temp)

    url = f"{_base_url()}/v1/chat/completions"
    timeout = float(os.environ.get("GMI_REQUEST_TIMEOUT_S", "30"))
    response = requests.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"GMI reasoning HTTP {response.status_code}: {response.text[:200]}")

    body = response.json()
    if body.get("error"):
        raise RuntimeError(f"GMI reasoning error: {body['error']}")
    content = body["choices"][0]["message"]["content"]
    data = _extract_json(content)

    urgency = str(data.get("urgency", "Medium")).title()
    if urgency not in _VALID_URGENCY:
        urgency = "Medium"

    return {
        "urgency": urgency,
        "risk_level": urgency,
        "situation": str(data.get("situation", "Possible hazard detected.")).strip(),
        "safest_next_action": str(data.get("safest_next_action", "Move away from the hazard.")).strip(),
        "voice_guidance": str(data.get("voice_guidance", "Move away from the hazard and find a safer area.")).strip(),
        "reasoning_summary": str(data.get("reasoning_summary", "")).strip(),
        "uncertainty": str(data.get("uncertainty", "")).strip(),
        "next_check": str(data.get("next_check", "")).strip(),
    }
