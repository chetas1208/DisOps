"""Voice router — GMI TTS primary, ElevenLabs fallback."""

from __future__ import annotations

from typing import Literal

from voice_output import speak as speak_elevenlabs
from voice_output_gmi import speak as speak_gmi
from voice_output_gmi import stop as stop_gmi

Backend = Literal["gmi", "elevenlabs_fallback"]
_last_backend: Backend | None = None


def last_backend() -> Backend | None:
    """Backend used by the most recent speak_with_fallback call."""
    return _last_backend


def speak_with_fallback(text: str, *, blocking: bool = True) -> None:
    global _last_backend

    cleaned = text.strip()
    if not cleaned:
        print("[voice router] WARNING: empty text — ignoring.", file=__import__("sys").stderr)
        return

    try:
        speak_gmi(cleaned, blocking=blocking)
        _last_backend = "gmi"
        print("[GMI TTS]")
    except Exception as exc:
        _last_backend = "elevenlabs_fallback"
        print(f"[ElevenLabs fallback] — {exc}")
        speak_elevenlabs(cleaned, blocking=blocking)


def stop_voice() -> None:
    """Halt any in-progress TTS playback."""
    stop_gmi()
    try:
        import sounddevice as sd

        sd.stop()
    except Exception:
        pass
