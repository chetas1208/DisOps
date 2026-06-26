"""GMI Cloud TTS — inworld-tts-2 via the request-queue API."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass, field

import numpy as np
import requests
import sounddevice as sd

DEFAULT_BASE_URL = "https://console.gmicloud.ai"
DEFAULT_MODEL = "inworld-tts-2"
DEFAULT_VOICE_ID = "Elizabeth"
REQUEST_TIMEOUT_S = 45.0
POLL_INTERVAL_S = 0.5
POLL_TIMEOUT_S = 40.0
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
    return os.environ.get("GMI_VIDEO_ENDPOINT_URL", DEFAULT_BASE_URL).strip().rstrip("/")


def _api_key() -> str:
    key = os.environ.get("GMI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GMI_API_KEY is not set. Export it before running:\n"
            "  export GMI_API_KEY='your-key-here'"
        )
    return key


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _next_session_id() -> int:
    global _session_counter
    _session_counter += 1
    return _session_counter


def _submit_tts(text: str) -> str:
    model = os.environ.get("GMI_TTS_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    voice_id = os.environ.get("GMI_TTS_VOICE_ID", DEFAULT_VOICE_ID).strip() or DEFAULT_VOICE_ID
    url = f"{_base_url()}/api/v1/ie/requestqueue/apikey/requests"
    payload = {"model": model, "payload": {"text": text, "voice_id": voice_id}}
    try:
        response = requests.post(url, headers=_headers(), json=payload, timeout=REQUEST_TIMEOUT_S)
    except requests.Timeout as exc:
        raise RuntimeError(f"GMI TTS submit timed out after {REQUEST_TIMEOUT_S}s") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"GMI TTS submit failed: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(f"GMI TTS HTTP {response.status_code}: {response.text[:300]}")

    body = response.json()
    request_id = body.get("request_id")
    if not request_id:
        raise RuntimeError(f"GMI TTS response missing request_id: {body!r}")
    return str(request_id)


def _poll_audio_url(request_id: str) -> str:
    url = f"{_base_url()}/api/v1/ie/requestqueue/apikey/requests/{request_id}"
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            response = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            raise RuntimeError(f"GMI TTS poll failed: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(f"GMI TTS poll HTTP {response.status_code}: {response.text[:300]}")

        body = response.json()
        status = str(body.get("status", "")).lower()
        if status == "success":
            outcome = body.get("outcome") or {}
            audio_url = outcome.get("audio_url")
            if not audio_url:
                media_urls = outcome.get("media_urls") or []
                if media_urls and isinstance(media_urls[0], dict):
                    audio_url = media_urls[0].get("url")
            if not audio_url:
                raise RuntimeError(f"GMI TTS success but no audio URL: {body!r}")
            return str(audio_url)
        if status in {"failed", "error", "cancelled"}:
            raise RuntimeError(f"GMI TTS job {status}: {body.get('error') or body}")

        time.sleep(POLL_INTERVAL_S)

    raise RuntimeError(f"GMI TTS timed out after {POLL_TIMEOUT_S}s (request {request_id})")


def _download_audio(audio_url: str) -> bytes:
    try:
        response = requests.get(audio_url, timeout=REQUEST_TIMEOUT_S)
    except requests.RequestException as exc:
        raise RuntimeError(f"GMI TTS audio download failed: {exc}") from exc
    if response.status_code != 200 or not response.content:
        raise RuntimeError(
            f"GMI TTS audio fetch HTTP {response.status_code}: {response.text[:200]}"
        )
    return response.content


def _decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    if audio_bytes[:4] == b"RIFF":
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            channels = wf.getnchannels()
            rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            if sampwidth != 2:
                raise RuntimeError(f"unsupported WAV sample width: {sampwidth}")
            frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
        return audio, rate

    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "24000",
                "pipe:1",
            ],
            input=audio_bytes,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to play GMI MP3 audio") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg decode failed: {exc.stderr.decode()[:200]}") from exc

    audio = np.frombuffer(proc.stdout, dtype=np.int16)
    return audio, 24000


def _play_audio(session: _PlaybackSession, audio_bytes: bytes) -> None:
    if os.environ.get("VOICE_PLAYBACK", "1").strip().lower() in {"0", "false", "no", "off"}:
        session.latency_ms = (time.perf_counter() - session.called_at) * 1000
        print(
            f"[voice/gmi] synthesis complete in {session.latency_ms:.0f}ms "
            f"(playback disabled, session {session.session_id})"
        )
        return

    audio, rate = _decode_audio(audio_bytes)
    if session.stop_event.is_set():
        session.interrupted = True
        return

    session.latency_ms = (time.perf_counter() - session.called_at) * 1000
    print(
        f"[voice/gmi] playback started in {session.latency_ms:.0f}ms "
        f"(session {session.session_id})"
    )

    try:
        sd.play(audio, samplerate=rate)
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Audio playback failed — no usable output device? ({exc})") from exc

    while True:
        stream = sd.get_stream()
        if stream is None or not stream.active:
            break
        if session.stop_event.is_set():
            sd.stop()
            session.interrupted = True
            break
        time.sleep(0.02)
    sd.stop()


def _synthesize_and_play(session: _PlaybackSession) -> None:
    request_id = _submit_tts(session.text)
    audio_url = _poll_audio_url(request_id)
    audio_bytes = _download_audio(audio_url)
    _play_audio(session, audio_bytes)


def _run_session(session: _PlaybackSession) -> None:
    try:
        _synthesize_and_play(session)
    except RuntimeError as exc:
        session.error = str(exc)
        print(f"[voice/gmi] ERROR (session {session.session_id}): {exc}", file=sys.stderr)
    except Exception as exc:
        session.error = f"Unexpected error: {exc}"
        print(f"[voice/gmi] ERROR (session {session.session_id}): {exc}", file=sys.stderr)
    finally:
        session.done_event.set()


def speak(text: str, *, blocking: bool = True) -> None:
    """Speak text via GMI inworld-tts-2 (interrupt-and-replace)."""
    cleaned = text.strip()
    if not cleaned:
        print("[voice/gmi] WARNING: speak() called with empty text — ignoring.", file=sys.stderr)
        return

    global _active_session

    with _lock:
        if _active_session is not None and not _active_session.done_event.is_set():
            print(
                f"[voice/gmi] interrupting session {_active_session.session_id} "
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
        name=f"gmi-voice-{session.session_id}",
        daemon=True,
    )
    worker.start()

    if not blocking:
        return

    while not session.done_event.wait(timeout=0.05):
        pass

    if session.error:
        raise RuntimeError(session.error)


def stop() -> None:
    """Stop any in-progress GMI TTS playback immediately."""
    global _active_session

    with _lock:
        if _active_session is not None and not _active_session.done_event.is_set():
            _active_session.stop_event.set()
            _active_session.interrupted = True

    try:
        sd.stop()
    except Exception:
        pass
