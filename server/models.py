"""Event schema for the DisOps pipeline.

A single ``PipelineEvent`` is the unit the backend streams to the browser over
the WebSocket. It is intentionally a superset of the shape described in the
product spec so the UI has everything it needs to render every panel.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Urgency = Literal["Low", "Medium", "High", "Critical"]
StageStatus = Literal["idle", "processing", "complete", "error"]

PIPELINE_STAGES: tuple[str, ...] = (
    "camera",
    "frame_sharding",
    "vision_models",
    "kimi_k2",
    "tts",
    "speaker",
)


def utc_now_iso() -> str:
    """ISO-8601 timestamp in UTC with a trailing Z, matching the spec."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clock_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _epoch() -> float:
    return time.time()


@dataclass
class StageState:
    """Status + timing for one node of the agent pipeline."""

    status: StageStatus = "idle"
    duration_ms: int | None = None
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "duration_ms": self.duration_ms, "label": self.label}


@dataclass
class PipelineEvent:
    """One analyzed frame's worth of agent output."""

    frame_id: str
    urgency: Urgency = "Low"
    scene_summary: str = "Monitoring environment…"
    situation: str = "No active incident."
    voice_guidance: str = "Monitoring your environment. I'll speak up if I detect danger."
    reasoning_summary: str = ""
    next_check: str = ""
    uncertainty: str = ""
    risk_level: str = "Low"
    safest_next_action: str = ""

    detected_hazards: list[str] = field(default_factory=list)
    detected_objects: list[str] = field(default_factory=list)
    people_detected: int = 0
    environment_conditions: list[str] = field(default_factory=list)
    motion_changes: str = "Stable"
    confidence: float = 0.5

    latency_ms: int = 0
    stage_durations: dict[str, int] = field(default_factory=dict)
    pipeline_status: dict[str, str] = field(default_factory=dict)

    scenario: str = "no_hazard"
    source: Literal["demo", "live"] = "demo"
    speak: bool = False
    timestamp: str = field(default_factory=utc_now_iso)
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LogEntry:
    """A single line in the live event timeline."""

    message: str
    level: Literal["info", "warn", "alert", "success"] = "info"
    time: str = field(default_factory=_clock_str)
    ts: float = field(default_factory=_epoch)

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message, "level": self.level, "time": self.time, "ts": self.ts}
