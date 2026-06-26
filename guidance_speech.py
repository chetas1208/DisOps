"""Fixed spoken phrases for evacuation guidance (GMI TTS)."""

from __future__ import annotations

from typing import Literal

Direction = Literal["left", "right", "straight", "stop"]

_VALID_DIRECTIONS: frozenset[str] = frozenset({"left", "right", "straight", "stop"})


def normalize_direction(direction: str | None) -> Direction | None:
    if direction in _VALID_DIRECTIONS:
        return direction  # type: ignore[return-value]
    return None


def spoken_guidance(hazard: bool, direction: str | None) -> str:
    """Short lines for TTS — clear vs hazard with direction."""
    if not hazard:
        return "All good."

    direction = normalize_direction(direction)
    phrases: dict[Direction, str] = {
        "right": "Take the staircase to the right, it looks clear.",
        "left": "Take the staircase to the left, it looks clear.",
        "straight": "Continue straight ahead, it looks clear.",
        "stop": "Stop here. Do not proceed.",
    }
    return phrases.get(direction or "right", "Take the nearest clear exit. It looks clear.")
