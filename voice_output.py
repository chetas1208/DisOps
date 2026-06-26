"""Voice output layer — ElevenLabs TTS with interrupt-and-replace playback."""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

import sounddevice as sd
from elevenlabs.client import ElevenLabs

# Calm, neutral narrator — not dramatic or robotic.
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George
MODEL_ID = "eleven_flash_v2_5"
OUTPUT_FORMAT = "pcm_16000"
SAMPLE_RATE = 16000

# Behavior when speak() is called while audio is already playing:
# interrupt the current utterance and replace it with the new one.
INTERRUPT_POLICY = "interrupt-and-replace"


@dataclass
class _PlaybackSession:
    session_id: int
    text: str
    called_at: float
    stop_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    interrupted: bool = False
    latency_ms: float | None = None
    error: str | None = None


_lock = threading.Lock()
_session_counter = 0
_active_session: _PlaybackSession | None = None


def _require_api_key() -> str:
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ELEVENLABS_API_KEY is not set. Export it before running:\n"
            "  export ELEVENLABS_API_KEY='your-key-here'"
        )
    return api_key


def _next_session_id() -> int:
    global _session_counter
    _session_counter += 1
    return _session_counter


def _iter_audio_chunks(client: ElevenLabs, text: str) -> Iterator[bytes]:
    stream = client.text_to_speech.stream(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        output_format=OUTPUT_FORMAT,
        optimize_streaming_latency=4,
        voice_settings={
            "stability": 0.65,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    )
    for chunk in stream:
        if isinstance(chunk, bytes) and chunk:
            yield chunk


def _play_pcm_stream(
    session: _PlaybackSession,
    chunks: Iterator[bytes],
) -> None:
    """Stream PCM chunks to the default output device until done or interrupted."""
    output_stream: sd.RawOutputStream | None = None
    playback_started = False

    try:
        for chunk in chunks:
            if session.stop_event.is_set():
                session.interrupted = True
                break

            if not playback_started:
                try:
                    output_stream = sd.RawOutputStream(
                        samplerate=SAMPLE_RATE,
                        channels=1,
                        dtype="int16",
                        blocksize=0,
                    )
                    output_stream.start()
                except sd.PortAudioError as exc:
                    raise RuntimeError(
                        f"Audio playback failed — no usable output device? ({exc})"
                    ) from exc

                session.latency_ms = (time.perf_counter() - session.called_at) * 1000
                playback_started = True
                print(
                    f"[voice] playback started in {session.latency_ms:.0f}ms "
                    f"(session {session.session_id})"
                )

            if session.stop_event.is_set():
                session.interrupted = True
                break

            output_stream.write(chunk)

    finally:
        if output_stream is not None:
            try:
                output_stream.stop()
                output_stream.close()
            except Exception:
                pass
        session.done_event.set()


def _run_session(session: _PlaybackSession) -> None:
    try:
        client = ElevenLabs(api_key=_require_api_key())
        chunks = _iter_audio_chunks(client, session.text)
        _play_pcm_stream(session, chunks)
    except RuntimeError as exc:
        session.error = str(exc)
        print(f"[voice] ERROR (session {session.session_id}): {exc}", file=sys.stderr)
    except Exception as exc:
        session.error = f"Unexpected error: {exc}"
        print(
            f"[voice] ERROR (session {session.session_id}): {exc}",
            file=sys.stderr,
        )
    finally:
        session.done_event.set()


def speak(text: str, *, blocking: bool = True) -> None:
    """
    Convert text to speech via ElevenLabs and play through the default audio device.

    Blocking (default): returns after playback finishes or this utterance is interrupted.
    Non-blocking (blocking=False): returns immediately; playback continues in the background.

    Interrupt policy (interrupt-and-replace):
    If speak() is called while audio is already playing, the current utterance is
    stopped and replaced by the new one. Debounce should make this rare.
    """
    cleaned = text.strip()
    if not cleaned:
        print("[voice] WARNING: speak() called with empty text — ignoring.", file=sys.stderr)
        return

    _require_api_key()

    global _active_session

    with _lock:
        if _active_session is not None and not _active_session.done_event.is_set():
            print(
                f"[voice] interrupting session {_active_session.session_id} "
                f"({INTERRUPT_POLICY})"
            )
            _active_session.stop_event.set()
            _active_session.interrupted = True

        session = _PlaybackSession(
            session_id=_next_session_id(),
            text=cleaned,
            called_at=time.perf_counter(),
        )
        _active_session = session

    worker = threading.Thread(
        target=_run_session,
        args=(session,),
        name=f"voice-session-{session.session_id}",
        daemon=True,
    )
    worker.start()

    if not blocking:
        return

    while not session.done_event.wait(timeout=0.05):
        pass

    if session.error:
        raise RuntimeError(session.error)


# ---------------------------------------------------------------------------
# Test harness — replace simulated decisions with live vision_core output later.
# ---------------------------------------------------------------------------
def _run_speak_latency_test(label: str, text: str) -> float | None:
    print(f"\n--- speak test: {label} ---")
    print(f"text: {text!r}")
    called_at = time.perf_counter()
    try:
        speak(text, blocking=True)
    except RuntimeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return None

    with _lock:
        session = _active_session

    if session and session.latency_ms is not None:
        print(f"measured start latency: {session.latency_ms:.0f}ms")
        return session.latency_ms

    print("WARNING: playback latency was not recorded.", file=sys.stderr)
    return (time.perf_counter() - called_at) * 1000


def _run_interrupt_test() -> None:
    print("\n--- speak test: back-to-back interrupt ---")
    first = "First alert, crowd forming near the east corridor."
    second = "Updated guidance, please proceed right toward the clear exit."

    first_session: _PlaybackSession | None = None

    def _start_first() -> None:
        try:
            speak(first, blocking=True)
        except RuntimeError as exc:
            print(f"first speak failed: {exc}", file=sys.stderr)

    starter = threading.Thread(target=_start_first, daemon=True)
    starter.start()

    # Wait until the first utterance owns the active session, then capture that object.
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline and first_session is None:
        with _lock:
            candidate = _active_session
            if candidate is not None and not candidate.done_event.is_set():
                first_session = candidate
                break
        time.sleep(0.01)

    if first_session is None:
        print("WARNING: first session never started — interrupt test inconclusive.", file=sys.stderr)
        starter.join(timeout=30)
        return

    try:
        speak(second, blocking=True)
    except RuntimeError as exc:
        print(f"second speak failed: {exc}", file=sys.stderr)

    starter.join(timeout=30)

    with _lock:
        active = _active_session

    if first_session.interrupted:
        print("interrupt-and-replace: first utterance was interrupted as expected")
    else:
        print(
            "WARNING: first utterance may not have been interrupted "
            "(first call may have finished before second started).",
            file=sys.stderr,
        )

    if active and active.text == second:
        print("final spoken text matches the replacement utterance")
    print("back-to-back test completed without crash")


def _run_pipeline_test() -> None:
    """Simulated frames → debounce → speak(). Plug analyze_hallway() in place of decisions."""
    from debounce import GuidanceDebouncer

    print("\n=== End-to-end pipeline (simulated decisions → debounce → speak) ===")

    # Later: decision = analyze_hallway(frame_path)
    simulated_frames = [
        {"hazard": False, "direction": None, "text": "clear, continue."},
        {"hazard": False, "direction": None, "text": "clear, continue."},
        {
            "hazard": True,
            "direction": "right",
            "text": "Crowd forming ahead, proceed right.",
        },
        {
            "hazard": True,
            "direction": "right",
            "text": "Crowd still ahead, proceed right.",
        },
        {"hazard": False, "direction": None, "text": "Path clear again, continue."},
    ]

    debouncer = GuidanceDebouncer()
    spoken = 0

    for index, decision in enumerate(simulated_frames):
        result = debouncer.process(decision)
        print(
            f"  frame {index}: hazard={decision['hazard']} "
            f"direction={decision['direction']} -> speak={result['speak']}"
        )
        if result["speak"] and result["text"]:
            spoken += 1
            print(f"    speaking: {result['text']!r}")
            try:
                speak(result["text"], blocking=True)
            except RuntimeError as exc:
                print(f"    speak failed: {exc}", file=sys.stderr)
                return

    print(f"pipeline complete — spoke {spoken} time(s) across {len(simulated_frames)} frames")


def main() -> int:
    try:
        _require_api_key()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    latencies: list[float] = []

    for label, text in (
        ("short clear", "Clear ahead, continue."),
        (
            "long hazard",
            "Crowd forming near the east corridor. Please proceed right toward the clear exit.",
        ),
    ):
        latency = _run_speak_latency_test(label, text)
        if latency is not None:
            latencies.append(latency)

    _run_interrupt_test()
    _run_pipeline_test()

    if latencies:
        print("\n=== Latency summary ===")
        for i, ms in enumerate(latencies, start=1):
            status = "OK" if ms <= 2000 else "SLOW"
            print(f"  test {i}: {ms:.0f}ms [{status}]")
        worst = max(latencies)
        best = min(latencies)
        print(f"  best: {best:.0f}ms  worst: {worst:.0f}ms  target: <=2000ms to first audio")

        if worst > 2000:
            print(
                "\nLatency is above target. Try:\n"
                "  - Keep eleven_flash_v2_5 (already the fastest model)\n"
                "  - Use pcm_16000 output (already set) — avoid mp3 decode delay\n"
                "  - Shorten spoken text where possible\n"
                "  - Run on a low-latency network / disable VPN\n"
                "  - Set optimize_streaming_latency=4 (already max)\n"
                "  - Pre-warm with a throwaway speak() call at startup\n"
                "  - Consider ElevenLabs Turbo/Flash streaming WebSocket if still slow",
                file=sys.stderr,
            )
            return 2

        print("\nLatency target met (<=2000ms to first audio).")
        return 0

    print("No latency measurements recorded — speak tests may have failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
