"""Vision reasoning core for evacuation guidance — single image in, structured decision out."""

from __future__ import annotations

import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Literal, TypedDict

from google import genai
from google.genai import types
from google.genai.errors import APIError

PROMPT = (
    "You are monitoring a hallway via a security camera for an evacuation guidance system. "
    "Describe what you see in one sentence. If there is a hazard (crowd, smoke, blockage, "
    "debris, fire), say so and recommend a direction (left/right/straight/stop) toward the "
    "clearest visible path. If the path is clear, say 'clear, continue.' "
    "Be concise — one short sentence only."
)

MODEL = "gemini-2.5-flash-lite"
MAX_SPOKEN_CHARS = 180

Direction = Literal["left", "right", "straight", "stop"]


class GuidanceResult(TypedDict):
    hazard: bool
    direction: Direction | None
    text: str


CLEAR_PATTERNS = (
    re.compile(r"\bclear\b.*\bcontinue\b", re.I),
    re.compile(r"\bpath\s+is\s+clear\b", re.I),
    re.compile(r"\ball\s+clear\b", re.I),
    re.compile(r"\bno\s+(visible\s+)?hazard\b", re.I),
)

HAZARD_PATTERNS = (
    re.compile(
        r"\b(crowd|smoke|blockage|blocked|blocking|debris|fire|flames|hazard|"
        r"obstruction|obstructed|commotion|chaos|people|person|group)\b",
        re.I,
    ),
)

DIRECTION_PATTERNS: list[tuple[re.Pattern[str], Direction]] = [
    (re.compile(r"\bturn\s+right\b|\bgo\s+right\b|\bhead\s+right\b|\bto\s+the\s+right\b|\bright\b", re.I), "right"),
    (re.compile(r"\bturn\s+left\b|\bgo\s+left\b|\bhead\s+left\b|\bto\s+the\s+left\b|\bleft\b", re.I), "left"),
    (re.compile(r"\bgo\s+straight\b|\bcontinue\s+straight\b|\bstraight\s+ahead\b|\bstraight\b", re.I), "straight"),
    (re.compile(r"\bstop\b|\bhalt\b|\bdo\s+not\s+proceed\b|\bwait\b", re.I), "stop"),
]


def _require_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it before running:\n"
            "  export GEMINI_API_KEY='your-key-here'"
        )
    return api_key


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    match = re.match(r"^(.+?[.!?])(?:\s|$)", text)
    sentence = match.group(1) if match else text
    if len(sentence) > MAX_SPOKEN_CHARS:
        sentence = sentence[: MAX_SPOKEN_CHARS - 3].rstrip() + "..."
    return sentence


def _parse_guidance(raw_text: str) -> GuidanceResult:
    text = _first_sentence(raw_text)
    lowered = text.lower()

    is_clear = any(pattern.search(lowered) for pattern in CLEAR_PATTERNS)
    has_hazard = any(pattern.search(lowered) for pattern in HAZARD_PATTERNS)

    direction: Direction | None = None
    for pattern, value in DIRECTION_PATTERNS:
        if pattern.search(lowered):
            direction = value
            break

    if is_clear and not has_hazard:
        return {"hazard": False, "direction": None, "text": text or "clear, continue."}

    if has_hazard or direction is not None:
        if has_hazard and direction is None:
            direction = "stop"
        return {"hazard": True, "direction": direction, "text": text}

    # No hazard language and no direction — treat as clear.
    return {"hazard": False, "direction": None, "text": text or "clear, continue."}


def analyze_hallway(image_path: str | Path) -> GuidanceResult:
    """
    Analyze a hallway camera frame and return evacuation guidance.

    Returns:
        {"hazard": bool, "direction": "left"|"right"|"straight"|"stop"|None, "text": str}
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    image_bytes = path.read_bytes()

    client = genai.Client(api_key=_require_api_key())

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=64,
            ),
        )
    except APIError as exc:
        raise RuntimeError(f"Gemini API call failed for {path.name}: {exc}") from exc

    raw_text = (response.text or "").strip()
    if not raw_text:
        raise RuntimeError(
            f"Gemini returned an empty response for {path.name}. "
            "Check model access, quota, or safety filters."
        )

    return _parse_guidance(raw_text)


def _iter_test_images(folder: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    images = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )
    return images


def main() -> int:
    test_dir = Path(__file__).resolve().parent / "test_images"
    if not test_dir.is_dir():
        print(f"ERROR: test_images folder not found at {test_dir}", file=sys.stderr)
        return 1

    images = _iter_test_images(test_dir)
    if not images:
        print(
            f"ERROR: No images found in {test_dir}. "
            "Add .jpg/.png files or run: python scripts/generate_test_images.py",
            file=sys.stderr,
        )
        return 1

    try:
        _require_api_key()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Analyzing {len(images)} image(s) with {MODEL}...\n")

    exit_code = 0
    for image_path in images:
        print(f"--- {image_path.name} ---")
        try:
            result = analyze_hallway(image_path)
            print(f"  hazard:    {result['hazard']}")
            print(f"  direction: {result['direction']}")
            print(f"  text:      {result['text']}")
        except (RuntimeError, FileNotFoundError, OSError) as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            exit_code = 1
        print()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
