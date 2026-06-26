"""Session orchestrator: owns live state, the demo loop, and WebSocket fan-out.

The orchestrator is the single source of truth for a monitoring session. It:

* tracks session flags (running / paused / muted / demo mode / active scenario),
* runs an async loop that emits one :class:`PipelineEvent` every 2-3 seconds in
  demo mode (simulating the full Camera → Vision → Kimi K2 → TTS pipeline),
* records a rolling event-timeline log,
* broadcasts every state change to all connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import asdict
from typing import Any

from fastapi import WebSocket

from guidance_speech import spoken_guidance

from voice_router import speak_with_fallback, stop_voice

from .config import frontend_config
from .models import LogEntry, PipelineEvent, utc_now_iso
from .pipeline import analyze_frame, vision_available
from .scenarios import DEFAULT_SCENARIO, get_scenario, scenario_list
from .voice import mark_voice_played, reset_voice, voice_cooldown_allows

# Descriptive pipeline_status values that match the product spec exactly.
_ACTIVE_STATUS = {
    "camera": "live",
    "frame_sharding": "complete",
    "vision_models": "complete",
    "kimi_k2": "reasoning_complete",
    "tts": "streaming",
    "speaker": "playing",
}
_IDLE_STATUS = {key: "idle" for key in _ACTIVE_STATUS}

_LOG_LIMIT = 60


class SessionManager:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._loop_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

        self.running = False
        self.paused = False
        self.muted = False
        self.demo_mode = False
        self.scenario = DEFAULT_SCENARIO
        self._frame_index = 0
        self._scenario_step = 0
        self._seq = 0

        self.last_event: PipelineEvent | None = None
        self.log: list[LogEntry] = []

    # ----------------------------------------------------------------- clients
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        await self._send(websocket, {"type": "snapshot", "data": self.snapshot()})

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def _send(self, websocket: WebSocket, message: dict[str, Any]) -> None:
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:  # noqa: BLE001 - drop dead sockets silently
            self._clients.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        for websocket in list(self._clients):
            await self._send(websocket, message)

    # ------------------------------------------------------------------- state
    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "paused": self.paused,
            "muted": self.muted,
            "demo_mode": self.demo_mode,
            "scenario": self.scenario,
            "vision_backend_available": vision_available(),
            "frame_interval_s": 2.5,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status(),
            "config": frontend_config(),
            "scenarios": scenario_list(),
            "last_event": self.last_event.to_dict() if self.last_event else None,
            "log": [entry.to_dict() for entry in self.log],
        }

    async def _log_event(self, message: str, level: str = "info") -> None:
        entry = LogEntry(message=message, level=level)  # type: ignore[arg-type]
        self.log.append(entry)
        if len(self.log) > _LOG_LIMIT:
            self.log = self.log[-_LOG_LIMIT:]
        await self.broadcast({"type": "log", "data": entry.to_dict()})

    async def _emit_status(self) -> None:
        await self.broadcast({"type": "status", "data": self.status()})

    async def _emit_event(self, event: PipelineEvent) -> None:
        self.last_event = event
        await self.broadcast({"type": "event", "data": event.to_dict()})

    # --------------------------------------------------------------- controls
    async def start(self) -> None:
        async with self._lock:
            if self.running:
                return
            self.running = True
            self.paused = False
            self._frame_index = 0
            self._scenario_step = 0
        reset_voice()
        await self._log_event("Camera stream started", "success")
        await self._log_event("GMI vision backend ready", "info")
        await self._emit_status()
        self._ensure_loop()

    async def stop(self) -> None:
        async with self._lock:
            self.running = False
            self.paused = False
        await self._cancel_loop()
        stop_voice()
        reset_voice()
        await self._log_event("Session stopped", "warn")
        await self.broadcast({"type": "pipeline_reset", "data": _IDLE_STATUS})
        await self._emit_status()

    async def pause(self, paused: bool) -> None:
        self.paused = paused
        if paused:
            stop_voice()
        await self._log_event("Monitoring paused" if paused else "Monitoring resumed", "info")
        await self._emit_status()

    async def set_muted(self, muted: bool) -> None:
        self.muted = muted
        if muted:
            stop_voice()
        await self._log_event("Voice muted" if muted else "Voice unmuted", "info")
        await self._emit_status()

    async def set_demo_mode(self, enabled: bool) -> None:
        self.demo_mode = enabled
        reset_voice()
        await self._log_event(
            "Demo Mode enabled — simulating live frames" if enabled else "Demo Mode disabled — using live camera",
            "info",
        )
        await self._emit_status()

    async def set_scenario(self, scenario_id: str) -> None:
        self.scenario = scenario_id
        self._scenario_step = 0
        label = get_scenario(scenario_id)["label"]
        await self._log_event(f"Scenario selected: {label}", "info")
        await self._emit_status()
        if self.demo_mode and self.running:
            await self._tick()  # immediate feedback on selection

    async def clear_session(self) -> None:
        await self.stop()
        self.log = []
        self.last_event = None
        self._seq = 0
        await self.broadcast({"type": "cleared", "data": self.snapshot()})

    async def replay_last(self) -> None:
        if self.last_event is not None:
            await self.broadcast({"type": "replay", "data": self.last_event.to_dict()})
            await self._log_event("Replaying last guidance", "info")
            if not self.muted and self.last_event.voice_guidance.strip():
                self._play_voice(self.last_event, force=True)

    async def snapshot_capture(self) -> None:
        await self._log_event(f"Snapshot captured at {utc_now_iso()}", "success")

    # ------------------------------------------------------- live frame ingest
    async def ingest_frame(self, frame_data_url: str) -> PipelineEvent:
        if not self.running or self.paused or self.demo_mode:
            raise RuntimeError("session is not accepting live frames")
        self._frame_index += 1
        self._seq += 1
        event = await asyncio.to_thread(
            analyze_frame, frame_data_url, frame_index=self._frame_index
        )
        event.seq = self._seq
        await self._log_event(f"{event.frame_id} analyzed (live)", "info")
        await self._emit_event(event)
        self._play_voice(event)
        return event

    # ---------------------------------------------------------------- demo loop
    def _ensure_loop(self) -> None:
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._run_loop())

    async def _cancel_loop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._loop_task = None

    async def _run_loop(self) -> None:
        try:
            while self.running:
                if self.demo_mode and not self.paused:
                    await self._tick()
                await asyncio.sleep(random.uniform(2.0, 3.0))
        except asyncio.CancelledError:
            raise

    async def _tick(self) -> None:
        """Emit one demo frame: animate the pipeline, then publish the event."""
        scenario = get_scenario(self.scenario)
        frames = scenario["frames"]
        frame_template = frames[self._scenario_step % len(frames)]
        self._scenario_step += 1
        self._frame_index += 1
        self._seq += 1

        event = self._build_demo_event(frame_template)
        event.seq = self._seq

        await self._log_event(f"{event.frame_id} sharded → local vision models", "info")
        await self._animate_pipeline(event)

        if event.detected_hazards:
            await self._log_event(
                f"Hazard detected: {', '.join(event.detected_hazards)}",
                "alert" if event.urgency in ("High", "Critical") else "warn",
            )
        await self._log_event("Kimi K2 generated guidance", "info")
        await self._emit_event(event)

    def _infer_direction(self, template: dict[str, Any], hazard: bool) -> str | None:
        from guidance_speech import normalize_direction

        if not hazard:
            return None
        if raw := template.get("direction"):
            return normalize_direction(str(raw))
        action = str(template.get("safest_next_action", "")).lower()
        for direction in ("right", "left", "straight"):
            if direction in action:
                return direction  # type: ignore[return-value]
        if "stop" in action or "do not" in action:
            return "stop"
        return "right"

    def _play_voice(self, event: PipelineEvent, *, force: bool = False) -> None:
        if not self.running or self.muted or not event.voice_guidance.strip():
            return
        if not force:
            if self.demo_mode or not event.speak:
                return
            if not voice_cooldown_allows():
                return
        try:
            speak_with_fallback(event.voice_guidance, blocking=False)
            mark_voice_played()
        except Exception as exc:  # noqa: BLE001 - voice must not crash the session
            asyncio.create_task(self._log_event(f"Voice playback failed: {exc}", "warn"))
            return
        asyncio.create_task(self._log_event("Voice instruction streamed (GMI TTS)", "success"))

    def _build_demo_event(self, template: dict[str, Any]) -> PipelineEvent:
        durations = {
            "camera": random.randint(8, 18),
            "frame_sharding": random.randint(10, 25),
            "vision_models": random.randint(140, 260),
            "kimi_k2": random.randint(480, 760),
            "tts": random.randint(150, 280),
            "speaker": random.randint(40, 90),
        }
        latency = sum(durations.values())
        event = PipelineEvent(
            frame_id=f"frame_{self._frame_index:03d}",
            scenario=self.scenario,
            source="demo",
            latency_ms=latency,
            stage_durations=durations,
            pipeline_status=dict(_ACTIVE_STATUS),
            timestamp=utc_now_iso(),
        )
        for key, value in template.items():
            setattr(event, key, value)

        hazard = bool(event.detected_hazards)
        direction = self._infer_direction(template, hazard)
        event.voice_guidance = spoken_guidance(hazard, direction)
        # Demo loop cycles clear/hazard frames — never auto-play TTS (use Replay).
        event.speak = False
        return event

    async def _animate_pipeline(self, event: PipelineEvent) -> None:
        """Walk each stage from processing → complete so the UI graph pulses."""
        stages = list(event.stage_durations.keys())
        for index, stage in enumerate(stages):
            statuses = {s: "complete" for s in stages[:index]}
            statuses[stage] = "processing"
            for s in stages[index + 1:]:
                statuses[s] = "idle"
            await self.broadcast(
                {
                    "type": "pipeline_progress",
                    "data": {
                        "active": stage,
                        "statuses": statuses,
                        "duration_ms": event.stage_durations[stage],
                        "frame_id": event.frame_id,
                    },
                }
            )
            # Brief, capped delay so the animation reads without slowing the demo.
            await asyncio.sleep(min(event.stage_durations[stage], 220) / 1000)
        await self.broadcast(
            {
                "type": "pipeline_progress",
                "data": {
                    "active": None,
                    "statuses": {s: "complete" for s in stages},
                    "frame_id": event.frame_id,
                },
            }
        )


def event_to_json(event: PipelineEvent) -> str:
    return json.dumps(asdict(event))
