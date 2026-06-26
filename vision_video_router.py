"""Video-chunk vision — middle-frame extract → GMI GPT-5.5."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ffmpeg_utils import extract_middle_frame
from vision_core import GuidanceResult
from vision_core_gmi import analyze_hallway as analyze_hallway_gpt55

Backend = Literal["gmi"]
_last_backend: Backend | None = None
_last_decision: GuidanceResult | None = None


def last_backend() -> Backend | None:
    return _last_backend


def reset_state() -> None:
    global _last_decision
    _last_decision = None


def analyze_chunk_with_fallback(video_path: str | Path) -> GuidanceResult:
    """
    Extract one frame from a video chunk and analyze with GMI GPT-5.5.

    Plug frontend video chunks here:
        decision = analyze_chunk_with_fallback(uploaded_chunk_path)
        debouncer.process(decision)
    """
    global _last_backend, _last_decision

    name = Path(video_path).name
    frame_path = extract_middle_frame(video_path)
    try:
        result = analyze_hallway_gpt55(frame_path)
        _last_decision = result
        _last_backend = "gmi"
        print(f"[Gmi/{name}] (frame: {frame_path.name})")
        return result
    finally:
        frame_path.unlink(missing_ok=True)
