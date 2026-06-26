"""Server-side voice debounce for live and demo sessions."""

from __future__ import annotations

import os
import time

from debounce import GuidanceDebouncer
from guidance_speech import normalize_direction, spoken_guidance

_debouncer = GuidanceDebouncer()
_last_play_at: float = 0.0
_last_stable_key: tuple[bool, str | None] | None = None
_stable_count: int = 0

COOLDOWN_S = float(os.environ.get("VOICE_COOLDOWN_S", "15"))
STABLE_FRAMES = max(1, int(os.environ.get("VOICE_STABLE_FRAMES", "2")))


def reset_voice() -> None:
    global _last_play_at, _last_stable_key, _stable_count
    _debouncer.reset()
    _last_play_at = 0.0
    _last_stable_key = None
    _stable_count = 0


def voice_cooldown_allows() -> bool:
    """Block rapid repeat TTS (demo loops, flickering vision)."""
    return time.monotonic() - _last_play_at >= COOLDOWN_S


def mark_voice_played() -> None:
    global _last_play_at
    _last_play_at = time.monotonic()


def _stable_key(hazard: bool, direction: str | None) -> tuple[bool, str | None]:
    return hazard, normalize_direction(direction)


def evaluate_voice(hazard: bool, direction: str | None) -> tuple[str, bool]:
    """Return fixed spoken line and whether to speak (stable + debounced + cooldown)."""
    global _last_stable_key, _stable_count

    norm = normalize_direction(direction)
    line = spoken_guidance(hazard, norm)
    key = _stable_key(hazard, direction)

    if key != _last_stable_key:
        _last_stable_key = key
        _stable_count = 1
        return line, False

    _stable_count += 1
    if _stable_count < STABLE_FRAMES:
        return line, False

    result = _debouncer.process(
        {
            "hazard": hazard,
            "direction": norm,
            "text": line,
        }
    )
    should_speak = bool(result["speak"])
    if should_speak and not voice_cooldown_allows():
        should_speak = False
    return line, should_speak
