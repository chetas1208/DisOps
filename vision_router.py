"""Vision router — GMI primary, Nemotron secondary, Gemini fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from vision_core import GuidanceResult, analyze_hallway as analyze_hallway_gemini
from vision_core_gmi import analyze_hallway as analyze_hallway_gmi
from vision_core_nemotron import analyze_hallway as analyze_hallway_nemotron

Backend = Literal["gmi", "nemotron", "gemini_fallback"]
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
    """
    Try vision backends in order: GMI → Nemotron → Gemini.

    Each attempt is logged to stdout so you can see which backend ran.
    """
    global _last_backend
    name = Path(image_path).name
    errors: list[str] = []

    for backend, fn in (
        ("gmi", analyze_hallway_gmi),
        ("nemotron", analyze_hallway_nemotron),
    ):
        try:
            result = fn(image_path)
            _validate_result(result)
            _last_backend = backend
            print(f"[{backend.capitalize()}] {name}")
            return result
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
            print(f"[{backend.capitalize()} failed] {name} — {exc}")

    _last_backend = "gemini_fallback"
    print(f"[Gemini fallback] {name} — {'; '.join(errors)}")
    result = analyze_hallway_gemini(image_path)
    _validate_result(result)
    return result
