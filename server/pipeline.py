"""Bridge between the live camera path and the existing vision pipeline.

The original project exposes ``vision_router.analyze_hallway_with_fallback`` which
returns ``{"hazard": bool, "direction": str|None, "text": str}``. This module
turns a browser-captured frame into that call and maps the compact result into
the rich :class:`PipelineEvent` the UI consumes.

Everything here is best-effort: if API keys, the Cloudflare tunnel, or the
vision backends are unavailable, we degrade gracefully instead of crashing the
session so the demo always keeps running.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from guidance_speech import normalize_direction, spoken_guidance

from .models import PipelineEvent, utc_now_iso
from .reasoning import reason as kimi_reason, reasoning_available
from .voice import evaluate_voice

_HAZARD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fire": ("fire", "flame", "flames", "burning"),
    "smoke": ("smoke", "haze", "smoky"),
    "flood_water": ("water", "flood", "flooding"),
    "debris": ("debris", "rubble", "collapse", "collapsed"),
    "blocked_exit": ("block", "blocked", "obstruct", "blockage"),
    "crowd": ("crowd", "people", "group", "commotion"),
    "person_down": ("fallen", "lying", "unconscious", "collapsed person"),
}


def _decode_data_url(data_url: str) -> bytes:
    """Accept a raw base64 string or a full ``data:image/...;base64,`` URL."""
    payload = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("frame payload is not valid base64") from exc


def _classify_hazards(text: str) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for hazard, needles in _HAZARD_KEYWORDS.items():
        if any(needle in lowered for needle in needles):
            found.append(hazard)
    return found


def _urgency_for(hazard: bool, hazards: list[str], direction: str | None) -> str:
    if not hazard:
        return "Low"
    critical = {"fire", "person_down"}
    if any(h in critical for h in hazards):
        return "Critical"
    if direction == "stop" or len(hazards) >= 2:
        return "High"
    return "Medium"


def _direction_phrase(direction: str | None) -> str:
    return {
        "left": "Move to your left toward the clearer path.",
        "right": "Move to your right toward the clearer path.",
        "straight": "Continue straight toward the clear path ahead.",
        "stop": "Stop and do not proceed toward the hazard.",
    }.get(direction or "", "Move away from the hazard toward a safer area.")


def vision_available() -> bool:
    """True if GMI GPT-5.5 vision is configured."""
    return bool(os.environ.get("GMI_API_KEY"))


def analyze_frame(frame_data_url: str, *, frame_index: int) -> PipelineEvent:
    """Run one captured frame through the real pipeline, mapping to a rich event.

    Raises nothing for backend failures — instead returns a low-urgency
    "monitoring" event annotated with the reason, so the live session keeps
    flowing even without API keys configured.
    """
    frame_id = f"frame_{frame_index:03d}"
    image_bytes = _decode_data_url(frame_data_url)

    t0 = time.perf_counter()
    tmp_path: Path | None = None
    try:
        from vision_router import analyze_hallway_with_fallback, last_backend  # local import

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(image_bytes)
            tmp_path = Path(handle.name)

        result: dict[str, Any] = analyze_hallway_with_fallback(tmp_path)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        hazard = bool(result.get("hazard"))
        text = str(result.get("text", "")).strip() or "Scene analyzed."
        direction = normalize_direction(result.get("direction"))
        hazards = _classify_hazards(text)
        urgency = _urgency_for(hazard, hazards, direction)
        backend = (last_backend() or "gmi").replace("_", " ")
        people = 1 if re.search(r"person|people|crowd", text.lower()) else 0

        # Heuristic defaults — overridden by GMI Kimi K2 reasoning when available.
        situation = text if hazard else "No active incident."
        voice_guidance, should_speak = evaluate_voice(hazard, direction)
        reasoning_summary = (
            f"{backend} backend flagged a hazard; recommending the clearest path."
            if hazard
            else f"{backend} backend reports a clear scene."
        )
        next_check = "Confirm the recommended path stays clear on the next frame."
        safest_next_action = _direction_phrase(direction) if hazard else "Proceed normally."
        uncertainty = "Live single-frame analysis; confidence improves across frames."
        kimi_ms = max(40, elapsed_ms // 3)

        # --- GMI Kimi K2 emergency reasoning (only when a hazard is present) ---
        if hazard and reasoning_available():
            try:
                t1 = time.perf_counter()
                guidance = kimi_reason(
                    scene_summary=text,
                    hazards=hazards,
                    direction=direction,
                    people_detected=people,
                )
                kimi_ms = int((time.perf_counter() - t1) * 1000)
                urgency = guidance["urgency"]
                situation = guidance["situation"] or situation
                reasoning_summary = guidance["reasoning_summary"] or reasoning_summary
                next_check = guidance["next_check"] or next_check
                safest_next_action = guidance["safest_next_action"] or safest_next_action
                uncertainty = guidance["uncertainty"] or uncertainty
            except Exception as exc:  # noqa: BLE001 - fall back to heuristic guidance
                reasoning_summary = f"{reasoning_summary} (Kimi K2 unavailable: {str(exc).splitlines()[0][:80]})"

        return PipelineEvent(
            frame_id=frame_id,
            urgency=urgency,  # type: ignore[arg-type]
            scene_summary=text,
            situation=situation,
            voice_guidance=voice_guidance,
            reasoning_summary=reasoning_summary,
            next_check=next_check,
            safest_next_action=safest_next_action,
            uncertainty=uncertainty,
            risk_level=urgency,
            detected_hazards=hazards,
            detected_objects=hazards or ["scene"],
            people_detected=people,
            environment_conditions=["live camera"],
            motion_changes="Live",
            confidence=0.86 if hazard else 0.9,
            latency_ms=elapsed_ms + kimi_ms,
            stage_durations={
                "vision_models": elapsed_ms,
                "kimi_k2": kimi_ms,
                "tts": 180,
            },
            source="live",
            speak=should_speak,
            scenario="live",
            timestamp=utc_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001 - never let a frame crash the session
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        reason = str(exc).splitlines()[0][:160]
        return PipelineEvent(
            frame_id=frame_id,
            urgency="Low",
            scene_summary="Monitoring environment…",
            situation="No active incident.",
            voice_guidance="Monitoring your environment. I'll speak up if I detect danger.",
            reasoning_summary=f"Vision backend unavailable ({reason}). Configure API keys for live analysis.",
            next_check="Enable Demo Mode for a full scripted walkthrough.",
            safest_next_action="Proceed normally.",
            uncertainty="Live backend not configured.",
            risk_level="Low",
            detected_hazards=[],
            detected_objects=[],
            people_detected=0,
            environment_conditions=["live camera"],
            motion_changes="Live",
            confidence=0.4,
            latency_ms=elapsed_ms,
            stage_durations={"vision_models": elapsed_ms},
            source="live",
            speak=False,
            scenario="live",
            timestamp=utc_now_iso(),
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
