#!/usr/bin/env python3
"""Integration tests for Nemotron routers (vision + voice with fallback)."""

from __future__ import annotations

import sys
from pathlib import Path

import voice_output_nemotron as vo_nem
import voice_router as vr
import voice_output as vo_eleven
from vision_router import analyze_hallway_with_fallback, last_backend as vision_last_backend

ROOT = Path(__file__).resolve().parent
TEST_IMAGES_DIR = ROOT / "test_images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def _iter_test_images() -> list[Path]:
    if not TEST_IMAGES_DIR.is_dir():
        return []
    return sorted(
        p for p in TEST_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _latency_from_voice_backend() -> float | None:
    with vo_nem._lock:
        session = vo_nem._active_session
    if session and session.latency_ms is not None:
        return session.latency_ms
    with vo_eleven._lock:
        session = vo_eleven._active_session
    if session and session.latency_ms is not None:
        return session.latency_ms
    return None


def test_vision_router() -> dict[str, int]:
    print("\n" + "=" * 60)
    print("VISION ROUTER — analyze_hallway_with_fallback()")
    print("=" * 60)

    images = _iter_test_images()
    counts = {"gmi": 0, "nemotron": 0, "gemini_fallback": 0, "errors": 0}

    if not images:
        print(f"No images in {TEST_IMAGES_DIR}")
        counts["errors"] += 1
        return counts

    for image_path in images:
        print(f"\n--- {image_path.name} ---")
        try:
            result = analyze_hallway_with_fallback(image_path)
            backend = vision_last_backend()
            print(f"  backend: {backend}")
            print(f"  result:  {result}")
            if backend == "gmi":
                counts["gmi"] += 1
            elif backend == "nemotron":
                counts["nemotron"] += 1
            elif backend == "gemini_fallback":
                counts["gemini_fallback"] += 1
        except Exception as exc:
            counts["errors"] += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    return counts


def test_voice_router() -> dict[str, int | list[float]]:
    print("\n" + "=" * 60)
    print("VOICE ROUTER — speak_with_fallback()")
    print("=" * 60)

    phrases = (
        ("short clear", "Clear ahead, continue."),
        (
            "long hazard",
            "Crowd forming near the east corridor. Please proceed right toward the clear exit.",
        ),
    )

    counts: dict[str, int | list[float]] = {
        "nemotron": 0,
        "elevenlabs_fallback": 0,
        "errors": 0,
        "latencies_ms": [],
    }

    for label, text in phrases:
        print(f"\n--- {label} ---")
        print(f"text: {text!r}")
        try:
            vr.speak_with_fallback(text, blocking=True)
            backend = vr.last_backend()
            latency = _latency_from_voice_backend()
            print(f"  backend: {backend}")
            if latency is not None:
                print(f"  latency: {latency:.0f}ms")
                counts["latencies_ms"].append(latency)
            if backend == "nemotron":
                counts["nemotron"] += 1
            elif backend == "elevenlabs_fallback":
                counts["elevenlabs_fallback"] += 1
        except Exception as exc:
            counts["errors"] += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    return counts


def print_summary(vision_counts: dict[str, int], voice_counts: dict[str, int | list[float]]) -> int:
    print("\n" + "=" * 60)
    print("NEMOTRON INTEGRATION SUMMARY")
    print("=" * 60)

    v_total = (
        vision_counts.get("gmi", 0)
        + vision_counts["nemotron"]
        + vision_counts["gemini_fallback"]
    )
    voice_latencies = voice_counts.get("latencies_ms", [])
    avg_latency = (
        sum(voice_latencies) / len(voice_latencies) if voice_latencies else None
    )

    print("\nVision:")
    print(f"  GMI successes:           {vision_counts.get('gmi', 0)}")
    print(f"  Nemotron successes:      {vision_counts['nemotron']}")
    print(f"  Gemini fallbacks:        {vision_counts['gemini_fallback']}")
    print(f"  Errors:                  {vision_counts['errors']}")
    if v_total and vision_counts.get("gmi", 0):
        gmi_pct = 100 * vision_counts["gmi"] / v_total
        print(f"  GMI hit rate:            {gmi_pct:.0f}%")

    print("\nVoice:")
    print(f"  Nemotron successes:      {voice_counts['nemotron']}")
    print(f"  ElevenLabs fallbacks:    {voice_counts['elevenlabs_fallback']}")
    print(f"  Errors:                  {voice_counts['errors']}")
    if avg_latency is not None:
        print(f"  Avg playback latency:    {avg_latency:.0f}ms")

    reliable = (
        vision_counts["errors"] == 0
        and voice_counts["errors"] == 0
        and (
            vision_counts.get("gmi", 0) > 0
            or vision_counts["nemotron"] > 0
            or voice_counts["nemotron"] > 0
        )
    )

    if vision_counts["nemotron"] == 0 and vision_counts["gemini_fallback"] > 0:
        print(
            "\nNote: Vision always fell back — server /health reports Omni on mock(fallback). "
            "Load real weights (MODEL_BACKEND=vllm or nim) for Nemotron vision."
        )
    if voice_counts["nemotron"] > 0:
        print("\nVoice: Nemotron /v1/voice/respond is working (may use fallback TTS engine on server).")

    if reliable:
        print("\nOVERALL: Nemotron integration is operational (with fallback when needed).")
        return 0

    if vision_counts["errors"] or voice_counts["errors"]:
        print("\nOVERALL: NOT RELIABLE — errors occurred during router tests.")
        return 1

    print("\nOVERALL: Routers work but Nemotron never handled a call successfully.")
    return 2


def main() -> int:
    print("Nemotron router integration test")
    print(f"Project root: {ROOT}")
    vision_counts = test_vision_router()
    voice_counts = test_voice_router()
    return print_summary(vision_counts, voice_counts)


if __name__ == "__main__":
    raise SystemExit(main())
