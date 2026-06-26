"""ffmpeg helpers — extract a still frame from a short video chunk."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg to use video-chunk fallback."
        )
    return path


def _video_duration_seconds(video_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return max(float(result.stdout.strip()), 0.0)
    except ValueError:
        return 0.0


def extract_middle_frame(video_path: str | Path, output_path: str | Path | None = None) -> Path:
    """
    Extract one JPEG frame from the middle of a video chunk (for GPT-5.5 fallback).
    """
    _require_ffmpeg()
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    if output_path is None:
        output_path = Path(tempfile.mkdtemp(prefix="disops-frame-")) / f"{video.stem}_frame.jpg"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    duration = _video_duration_seconds(video)
    timestamp = max(duration / 2, 0.0) if duration > 0 else 0.0

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out.is_file():
        raise RuntimeError(
            f"ffmpeg frame extract failed for {video.name}: "
            f"{(result.stderr or result.stdout)[-500:]}"
        )
    return out
