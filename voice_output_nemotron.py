"""Nemotron voice backend — same contract as voice_output.speak."""

from __future__ import annotations

import io
import os
import sys
import threading
import time
import wave
from dataclasses import dataclass, field

import numpy as np
import requests
import sounddevice as sd

DEFAULT_BASE_URL = "https://vermont-utah-bidder-rod.trycloudflare.com"
REQUEST_TIMEOUT_S = 3.0
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


def _base_url() -> str:
    return os.environ.get("NEMOTRON_ENDPOINT_URL", DEFAULT_BASE_URL).strip().rstrip("/")


def _next_session_id() -> int:
    global _session_counter
    _session_counter += 1
    return _session_counter


def _synthesize_wav(text: str) -> bytes:
    url = f"{_base_url()}/v1/voice/respond"
    try:
        response = requests.post(
            url,
            data={"text": text, "wait": "true"},
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Nemotron voice timed out after {REQUEST_TIMEOUT_S}s"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Nemotron voice request failed: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Nemotron voice HTTP {response.status_code}: {response.text[:300]}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("Nemotron voice returned non-JSON") from exc

    if body.get("status") != "succeeded":
        raise RuntimeError(f"Nemotron voice job failed: {body.get('error') or body}")

    result = body.get("result") or {}
    audio_url = result.get("audio_url")
    if not audio_url:
        raise RuntimeError(f"Nemotron voice response missing audio_url: {body!r}")

    audio_resp = requests.get(f"{_base_url()}{audio_url}", timeout=REQUEST_TIMEOUT_S)
    if audio_resp.status_code != 200:
        raise RuntimeError(
            f"Nemotron audio fetch HTTP {audio_resp.status_code}: {audio_resp.text[:200]}"
        )
    if not audio_resp.content:
        raise RuntimeError("Nemotron audio fetch returned empty body")
    return audio_resp.content


def _play_wav_bytes(session: _PlaybackSession, wav_bytes: bytes) -> None:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            if sampwidth != 2:
                raise RuntimeError(f"unsupported WAV sample width: {sampwidth}")
            frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels)

        if session.stop_event.is_set():
            session.interrupted = True
            return

        session.latency_ms = (time.perf_counter() - session.called_at) * 1000
        print(
            f"[voice/nemotron] playback started in {session.latency_ms:.0f}ms "
            f"(session {session.session_id})"
        )

        try:
            sd.play(audio, samplerate=rate)
        except sd.PortAudioError as exc:
            raise RuntimeError(
                f"Audio playback failed — no usable output device? ({exc})"
            ) from exc

        while True:
            stream = sd.get_stream()
            if stream is None or not stream.active:
                break
            if session.stop_event.is_set():
                sd.stop()
                session.interrupted = True
                break
            time.sleep(0.02)
    finally:
        sd.stop()


def _run_session(session: _PlaybackSession) -> None:
    try:
        wav_bytes = _synthesize_wav(session.text)
        _play_wav_bytes(session, wav_bytes)
    except RuntimeError as exc:
        session.error = str(exc)
        print(f"[voice/nemotron] ERROR (session {session.session_id}): {exc}", file=sys.stderr)
    except Exception as exc:
        session.error = f"Unexpected error: {exc}"
        print(
            f"[voice/nemotron] ERROR (session {session.session_id}): {exc}",
            file=sys.stderr,
        )
    finally:
        session.done_event.set()


def speak(text: str, *, blocking: bool = True) -> None:
    """
    Speak text via Nemotron /v1/voice/respond (WAV playback).

    Interrupt policy (interrupt-and-replace): a new call stops any in-progress audio.
    """
    cleaned = text.strip()
    if not cleaned:
        print("[voice/nemotron] WARNING: speak() called with empty text — ignoring.", file=sys.stderr)
        return

    global _active_session

    with _lock:
        if _active_session is not None and not _active_session.done_event.is_set():
            print(
                f"[voice/nemotron] interrupting session {_active_session.session_id} "
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
        name=f"nemotron-voice-{session.session_id}",
        daemon=True,
    )
    worker.start()

    if not blocking:
        return

    while not session.done_event.wait(timeout=0.05):
        pass

    if session.error:
        raise RuntimeError(session.error)
