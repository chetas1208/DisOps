"""Vision router — GMI GPT-5.5 only."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from vision_core import GuidanceResult
from vision_core_gmi import analyze_hallway as analyze_hallway_gmi

Backend = Literal["gmi"]
_last_backend: Backend | None = None


def _validate_result(result: GuidanceResult) -> None:
    if not isinstance(result, dict):
        raise ValueError(f"expected dict, got {type(result).__name__}")
    for key in ("hazard", "direction", "text"):
        if key not in result:
            raise ValueError(f"missing key '{key}'")
    if not isinstance(result["hazard"], bool):
        raise ValueError("hazard must be bool")
    if not isinstance(result["text"], str) or not result["text"].strip():
        raise ValueError("text must be a non-empty string")
    if result["hazard"] and result["direction"] not in {
        "left", "right", "straight", "stop",
    }:
        raise ValueError(
            f"hazard=true requires direction left/right/straight/stop, got {result['direction']!r}"
        )


def last_backend() -> Backend | None:
    """Backend used by the most recent analyze_hallway_with_fallback call."""
    return _last_backend


def analyze_hallway_with_fallback(image_path: str | Path) -> GuidanceResult:
    """Analyze one still frame via GMI GPT-5.5 vision."""
    global _last_backend
    name = Path(image_path).name

    result = analyze_hallway_gmi(image_path)
    _validate_result(result)
    _last_backend = "gmi"
    print(f"[Gmi] {name}")
    return result
