#!/usr/bin/env python3
"""Test video-chunk pipeline: middle-frame extract → GMI GPT-5.5."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from debounce import GuidanceDebouncer
from vision_video_router import analyze_chunk_with_fallback, last_backend, reset_state

ROOT = Path(__file__).resolve().parent
TEST_VIDEOS_DIR = ROOT / "test_videos"
TEST_IMAGES_DIR = ROOT / "test_images"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}


def _ensure_test_videos() -> list[Path]:
    TEST_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        p for p in TEST_VIDEOS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if existing:
        return existing

    # Build 3s clips from hallway test images if none exist.
    images = sorted(TEST_IMAGES_DIR.glob("*.png"))[:3]
    if not images:
        print("No test images to synthesize videos from.", file=sys.stderr)
        return []

    out: list[Path] = []
    for image in images:
        target = TEST_VIDEOS_DIR / f"{image.stem}_3s.mp4"
        if target.is_file():
            out.append(target)
            continue
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", str(image),
            "-c:v", "libx264", "-t", "3", "-pix_fmt", "yuv420p", "-r", "5",
            str(target),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        out.append(target)
    return sorted(out)


def main() -> int:
    print("Video pipeline test")
    print(f"Root: {ROOT}\n")

    videos = _ensure_test_videos()
    if not videos:
        return 1

    reset_state()
    debouncer = GuidanceDebouncer()
    backends: dict[str, int] = {"gmi": 0, "errors": 0}
    spoke = 0

    for video in videos:
        print(f"--- {video.name} ---")
        t0 = time.perf_counter()
        try:
            decision = analyze_chunk_with_fallback(video)
            ms = (time.perf_counter() - t0) * 1000
            backend = last_backend()
            if backend:
                backends[backend] = backends.get(backend, 0) + 1

            debounce_result = debouncer.process(decision)
            print(f"  backend:   {backend}")
            print(f"  latency:   {ms:.0f}ms")
            print(f"  decision:  {decision}")
            print(f"  debounce:  speak={debounce_result['speak']}")
            if debounce_result["speak"]:
                spoke += 1
                print(f"  would say: {debounce_result['text']!r}")
        except Exception as exc:
            backends["errors"] += 1
            print(f"  ERROR: {exc}")
        print()

    print("=== Summary ===")
    print(f"  GMI GPT-5.5:      {backends.get('gmi', 0)}")
    print(f"  Errors:           {backends.get('errors', 0)}")
    print(f"  Debouncer spoke:  {spoke} time(s)")

    if backends.get("errors", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
