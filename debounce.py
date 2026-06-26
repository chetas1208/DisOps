"""Debounce layer for evacuation guidance — decides when to speak vs stay silent."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal, TypedDict

Direction = Literal["left", "right", "straight", "stop"]


class Decision(TypedDict):
    hazard: bool
    direction: Direction | None
    text: str


class SpeakResult(TypedDict):
    speak: bool
    text: str | None


class GuidanceDebouncer:
    """Stateful filter over vision_core decisions — only speaks when state meaningfully changes."""

    def __init__(self) -> None:
        self._last_spoken: Decision | None = None

    def process(self, decision: Decision) -> SpeakResult:
        """
        Evaluate one frame's decision and return whether to speak aloud.

        Plug real-time frames here later:
            result = debouncer.process(analyze_hallway(frame_path))
        """
        if self._last_spoken is None:
            self._last_spoken = deepcopy(decision)
            # Stay silent on a clear first frame — only speak when a hazard appears.
            if decision["hazard"]:
                return {"speak": True, "text": decision["text"]}
            return {"speak": False, "text": None}

        hazard_changed = decision["hazard"] != self._last_spoken["hazard"]
        direction_changed = (
            decision["hazard"]
            and decision["direction"] != self._last_spoken["direction"]
        )

        if hazard_changed or direction_changed:
            self._last_spoken = deepcopy(decision)
            return {"speak": True, "text": decision["text"]}

        return {"speak": False, "text": None}

    def reset(self) -> None:
        """Clear spoken-state (e.g. on session restart)."""
        self._last_spoken = None


def _decision(
    hazard: bool,
    direction: Direction | None,
    text: str,
) -> Decision:
    return {"hazard": hazard, "direction": direction, "text": text}


def _clear(text: str = "clear, continue.") -> Decision:
    return _decision(False, None, text)


def _hazard(direction: Direction, text: str) -> Decision:
    return _decision(True, direction, text)


# ---------------------------------------------------------------------------
# Simulated frame sequences — replace with live analyze_hallway() calls later.
# ---------------------------------------------------------------------------
TEST_SEQUENCES: dict[str, list[Decision]] = {
    "A": [
        _clear(),
        _clear(),
        _clear(),
        _hazard("right", "Crowd ahead, go right."),
        _hazard("right", "Crowd ahead, go right."),
        _hazard("right", "People blocking center, turn right."),
        _clear("Path clear again, continue."),
        _clear(),
    ],
    "B": [
        _clear(),
        _hazard("right", "Smoke ahead, go right."),
        _hazard("right", "Smoke ahead, go right."),
        _hazard("left", "Path blocked right, turn left."),
        _hazard("left", "Path blocked right, turn left."),
    ],
    "C": [
        _clear(),
        _clear(),
        _clear(),
        _clear(),
    ],
}

EXPECTED_SPEAK_INDICES: dict[str, list[int]] = {
    "A": [3, 6],
    "B": [1, 3],
    "C": [],
}


def _run_sequence(name: str, decisions: list[Decision]) -> list[int]:
    debouncer = GuidanceDebouncer()
    spoken_indices: list[int] = []

    print(f"=== Sequence {name} ===")
    for index, decision in enumerate(decisions):
        result = debouncer.process(decision)
        spoke = result["speak"]
        if spoke:
            spoken_indices.append(index)

        hazard = decision["hazard"]
        direction = decision["direction"]
        print(
            f"  [{index}] hazard={hazard!s:5} direction={str(direction):6} "
            f"-> speak={str(spoke):5}  text={result['text']!r}"
        )

    expected = EXPECTED_SPEAK_INDICES[name]
    match = spoken_indices == expected
    status = "PASS" if match else "FAIL"
    print(f"  Spoke at indices: {spoken_indices}  (expected {expected})  [{status}]\n")
    return spoken_indices


def main() -> int:
    all_match = True
    for name, decisions in TEST_SEQUENCES.items():
        spoken = _run_sequence(name, decisions)
        if spoken != EXPECTED_SPEAK_INDICES[name]:
            all_match = False

    if all_match:
        print("All sequences matched expected speak indices.")
        return 0

    print("One or more sequences did NOT match expected behavior.", file=__import__("sys").stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
