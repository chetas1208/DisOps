#!/usr/bin/env python3
"""End-to-end verification for the evacuation-guidance pipeline."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

from debounce import EXPECTED_SPEAK_INDICES, TEST_SEQUENCES, GuidanceDebouncer
from vision_core import analyze_hallway
import voice_output as vo
from voice_output import speak

ROOT = Path(__file__).resolve().parent
TEST_IMAGES_DIR = ROOT / "test_images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VALID_DIRECTIONS = frozenset({"left", "right", "straight", "stop"})
LATENCY_TARGET_MS = 2000

SECTION_RESULTS: dict[str, tuple[bool, str]] = {}


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def _set_section(section: str, passed: bool, reason: str = "OK") -> None:
    SECTION_RESULTS[section] = (passed, reason)


def _validate_vision_result(result: Any) -> list[str]:
    issues: list[str] = []

    if not isinstance(result, dict):
        return [f"expected dict, got {type(result).__name__}"]

    for key in ("hazard", "direction", "text"):
        if key not in result:
            issues.append(f"missing key '{key}'")

    if "hazard" in result and not isinstance(result["hazard"], bool):
        issues.append(f"'hazard' should be bool, got {type(result['hazard']).__name__}")

    if "text" in result and not isinstance(result["text"], str):
        issues.append(f"'text' should be str, got {type(result['text']).__name__}")

    direction = result.get("direction")
    if direction is not None and direction not in VALID_DIRECTIONS:
        issues.append(
            f"invalid direction {direction!r} (allowed: {sorted(VALID_DIRECTIONS)} or null)"
        )

    if result.get("hazard") is True:
        if direction not in VALID_DIRECTIONS:
            issues.append(
                f"hazard=true but direction is {direction!r} "
                f"(expected one of {sorted(VALID_DIRECTIONS)})"
            )

    return issues


def _iter_test_images() -> list[Path]:
    if not TEST_IMAGES_DIR.is_dir():
        return []
    return sorted(
        p for p in TEST_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _latency_from_last_session() -> float | None:
    with vo._lock:
        session = vo._active_session
    if session and session.latency_ms is not None:
        return session.latency_ms
    return None


def run_section_1() -> list[tuple[Path, dict[str, Any] | None, str | None]]:
    _header("=== SECTION 1: VISION CORE ===")

    images = _iter_test_images()
    if not images:
        reason = f"no images found in {TEST_IMAGES_DIR}"
        print(f"FAIL: {reason}")
        _set_section("SECTION 1", False, reason)
        return []

    results: list[tuple[Path, dict[str, Any] | None, str | None]] = []
    malformed = 0
    errors = 0

    for image_path in images:
        print(f"\n--- {image_path.name} ---")
        try:
            decision = analyze_hallway(image_path)
            print(f"  result: {decision}")
            issues = _validate_vision_result(decision)
            if issues:
                malformed += 1
                for issue in issues:
                    print(f"  ⚠ UNEXPECTED: {issue}")
            results.append((image_path, decision, None))
        except Exception as exc:
            errors += 1
            err = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {err}")
            results.append((image_path, None, err))

    if errors:
        reason = f"{errors}/{len(images)} image(s) raised errors"
        print(f"\nSECTION 1 result: FAIL — {reason}")
        _set_section("SECTION 1", False, reason)
    elif malformed:
        reason = f"{malformed}/{len(images)} image(s) returned malformed output"
        print(f"\nSECTION 1 result: FAIL — {reason}")
        _set_section("SECTION 1", False, reason)
    else:
        print(f"\nSECTION 1 result: PASS — {len(images)} image(s) analyzed successfully")
        _set_section("SECTION 1", True)

    return results


def run_section_2() -> None:
    _header("=== SECTION 2: DEBOUNCE LOGIC ===")

    all_pass = True
    for name, decisions in TEST_SEQUENCES.items():
        debouncer = GuidanceDebouncer()
        spoken_indices: list[int] = []

        print(f"\n--- Sequence {name} ---")
        for index, decision in enumerate(decisions):
            result = debouncer.process(decision)
            spoke = result["speak"]
            if spoke:
                spoken_indices.append(index)
            print(
                f"  [{index}] hazard={decision['hazard']!s:5} "
                f"direction={str(decision['direction']):6} "
                f"-> speak={str(spoke):5}  text={result['text']!r}"
            )

        expected = EXPECTED_SPEAK_INDICES[name]
        match = spoken_indices == expected
        status = "PASS" if match else "FAIL"
        print(f"  Spoke at indices: {spoken_indices}  (expected {expected})  [{status}]")
        if not match:
            all_pass = False

    if all_pass:
        print("\nSECTION 2 result: PASS — all 3 sequences matched expected speak indices")
        _set_section("SECTION 2", True)
    else:
        reason = "one or more sequences did not match expected speak indices"
        print(f"\nSECTION 2 result: FAIL — {reason}")
        _set_section("SECTION 2", False, reason)


def run_section_3() -> None:
    _header("=== SECTION 3: VOICE OUTPUT ===")

    phrases = (
        ("short clear", "Clear ahead, continue."),
        (
            "long hazard",
            "Crowd forming near the east corridor. Please proceed right toward the clear exit.",
        ),
    )

    latencies: list[float] = []
    errors = 0

    for label, text in phrases:
        print(f"\n--- {label} ---")
        print(f"text: {text!r}")
        try:
            speak(text, blocking=True)
            latency = _latency_from_last_session()
            if latency is None:
                errors += 1
                print("  ERROR: latency was not recorded by voice_output")
            else:
                latencies.append(latency)
                status = "PASS" if latency <= LATENCY_TARGET_MS else "FAIL"
                print(f"  latency: {latency:.0f}ms  [{status} (target <= {LATENCY_TARGET_MS}ms)]")
        except Exception as exc:
            errors += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    if errors:
        reason = f"{errors} speak call(s) failed or did not record latency"
        print(f"\nSECTION 3 result: FAIL — {reason}")
        _set_section("SECTION 3", False, reason)
        return

    slow = [ms for ms in latencies if ms > LATENCY_TARGET_MS]
    if slow:
        worst = max(slow)
        reason = f"latency over {LATENCY_TARGET_MS}ms (worst: {worst:.0f}ms)"
        print(f"\nSECTION 3 result: FAIL — {reason}")
        _set_section("SECTION 3", False, reason)
        return

    avg = sum(latencies) / len(latencies)
    print(
        f"\nSECTION 3 result: PASS — both phrases under {LATENCY_TARGET_MS}ms "
        f"(latencies: {[f'{ms:.0f}ms' for ms in latencies]}, avg {avg:.0f}ms)"
    )
    _set_section("SECTION 3", True)


def run_section_4(
    vision_results: list[tuple[Path, dict[str, Any] | None, str | None]],
) -> None:
    _header("=== SECTION 4: FULL PIPELINE INTEGRATION ===")

    usable = [(path, decision) for path, decision, err in vision_results if decision is not None]
    if not usable:
        reason = "no successful vision results from Section 1 to feed into pipeline"
        print(f"FAIL: {reason}")
        _set_section("SECTION 4", False, reason)
        return

    debouncer = GuidanceDebouncer()
    total_images = 0
    total_spoke = 0
    speech_latencies: list[float] = []
    errors = 0

    print("Timeline (vision → debounce → speak):\n")

    for image_path, decision in usable:
        total_images += 1
        debounce_result = debouncer.process(decision)
        spoke = debounce_result["speak"]
        latency_ms: float | None = None

        line = (
            f"  [{image_path.name}] → "
            f"vision={decision} → "
            f"debounce={{speak: {spoke}, text: {debounce_result['text']!r}}} → "
        )

        if spoke and debounce_result["text"]:
            try:
                speak(debounce_result["text"], blocking=True)
                latency_ms = _latency_from_last_session()
                total_spoke += 1
                if latency_ms is not None:
                    speech_latencies.append(latency_ms)
                line += f"spoke: yes, latency={latency_ms:.0f}ms" if latency_ms else "spoke: yes, latency=unknown"
            except Exception as exc:
                errors += 1
                line += f"spoke: ERROR ({type(exc).__name__}: {exc})"
        else:
            line += "spoke: no"

        print(line)

    print("\n--- Section 4 summary ---")
    print(f"  total images processed: {total_images}")
    print(f"  total times spoke:      {total_spoke}")
    if speech_latencies:
        avg_latency = sum(speech_latencies) / len(speech_latencies)
        print(f"  average speech latency: {avg_latency:.0f}ms")
    else:
        avg_latency = None
        print("  average speech latency: n/a (no successful speech calls)")

    if errors:
        reason = f"{errors} speech call(s) failed during pipeline integration"
        print(f"\nSECTION 4 result: FAIL — {reason}")
        _set_section("SECTION 4", False, reason)
        return

    print("\nSECTION 4 result: PASS — pipeline ran end-to-end without errors")
    _set_section("SECTION 4", True)


def print_final_summary() -> int:
    _header("=== FINAL SUMMARY ===")

    for section in ("SECTION 1", "SECTION 2", "SECTION 3", "SECTION 4"):
        passed, reason = SECTION_RESULTS.get(section, (False, "not run"))
        status = "PASS" if passed else "FAIL"
        suffix = "" if passed else f" — {reason}"
        print(f"  {section}: {status}{suffix}")

    all_passed = all(
        SECTION_RESULTS.get(section, (False,))[0]
        for section in ("SECTION 1", "SECTION 2", "SECTION 3", "SECTION 4")
    )

    if all_passed:
        print("\n  OVERALL: READY FOR INTEGRATION")
        print("  All pipeline stages verified successfully.")
        return 0

    failed = [
        f"{section} ({reason})"
        for section in ("SECTION 1", "SECTION 2", "SECTION 3", "SECTION 4")
        for ok, reason in [SECTION_RESULTS.get(section, (False, "not run"))]
        if not ok
    ]
    print("\n  OVERALL: NOT READY")
    print(f"  Blocked by: {', '.join(failed)}")
    return 1


def main() -> int:
    print("Evacuation guidance — full pipeline test")
    print(f"Project root: {ROOT}")

    try:
        vision_results = run_section_1()
        run_section_2()
        run_section_3()
        run_section_4(vision_results)
        return print_final_summary()
    except Exception:
        print("\nUnexpected error in test_full_pipeline.py:", file=sys.stderr)
        traceback.print_exc()
        _header("=== FINAL SUMMARY ===")
        print("  OVERALL: NOT READY")
        print("  Blocked by: test script crashed (see traceback above)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
