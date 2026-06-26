"""Video-chunk vision router — Omni video → Kimi reasoning, GPT-5.5 frame fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ffmpeg_utils import extract_middle_frame
from vision_core import GuidanceResult
from vision_core_gmi import analyze_hallway as analyze_hallway_gpt55
from vision_core_omni_video import analyze_video_chunk
from vision_reasoning_gmi import decide_from_omni

Backend = Literal["omni_kimi", "gpt55_fallback"]
_last_backend: Backend | None = None
_last_omni_summary: str | None = None
_last_decision: GuidanceResult | None = None


def last_backend() -> Backend | None:
    return _last_backend


def last_omni_summary() -> str | None:
    return _last_omni_summary


def reset_state() -> None:
    global _last_omni_summary, _last_decision
    _last_omni_summary = None
    _last_decision = None


def analyze_chunk_with_fallback(video_path: str | Path) -> GuidanceResult:
    """
    Primary: 3s video chunk → Nemotron Omni → GMI Kimi → {hazard, direction, text}
    Fallback: ffmpeg middle frame → GMI GPT-5.5 image analysis

    Plug frontend video chunks here:
        decision = analyze_chunk_with_fallback(uploaded_chunk_path)
        debouncer.process(decision)
    """
    global _last_backend, _last_omni_summary, _last_decision

    name = Path(video_path).name

    try:
        omni_text = analyze_video_chunk(video_path)
        result = decide_from_omni(
            omni_text,
            prior_omni_text=_last_omni_summary,
            prior_decision=_last_decision,
        )
        _last_omni_summary = omni_text
        _last_decision = result
        _last_backend = "omni_kimi"
        print(f"[Omni+Kimi] {name}")
        return result
    except Exception as omni_exc:
        print(f"[Omni+Kimi failed] {name} — {omni_exc}")
        try:
            frame_path = extract_middle_frame(video_path)
            result = analyze_hallway_gpt55(frame_path)
            _last_decision = result
            _last_backend = "gpt55_fallback"
            print(f"[GPT-5.5 fallback] {name} (frame: {frame_path.name})")
            return result
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Video pipeline failed for {name}. "
                f"Omni: {omni_exc} | Fallback: {fallback_exc}"
            ) from fallback_exc
