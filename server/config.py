"""Runtime UI config — single source of truth for labels the frontend displays."""

from __future__ import annotations

import os

from .models import PIPELINE_STAGES


def _vision_model() -> str:
    return os.environ.get("GMI_VISION_MODEL", "openai/gpt-5.5").strip() or "openai/gpt-5.5"


def _reasoning_model() -> str:
    return os.environ.get("GMI_REASONING_MODEL", "").strip()


def _tts_model() -> str:
    return os.environ.get("GMI_TTS_MODEL", "inworld-tts-2").strip() or "inworld-tts-2"


def vision_model_label() -> str:
    model = _vision_model().lower()
    if "gpt-5" in model:
        return "GPT-5.5"
    return _vision_model().split("/")[-1]


def reasoning_model_label() -> str:
    model = _reasoning_model().lower()
    if "kimi" in model:
        return "Kimi K2"
    if model:
        return _reasoning_model().split("/")[-1]
    return "Reasoning"


def pipeline_stages() -> list[dict[str, str]]:
    labels: dict[str, str] = {
        "camera": "Camera Capture",
        "frame_sharding": "Frame Sharding",
        "vision_models": f"Vision ({vision_model_label()})",
        "kimi_k2": f"Reasoning ({reasoning_model_label()})",
        "tts": "Voice / TTS",
        "speaker": "Speaker Output",
    }
    icons: dict[str, str] = {
        "camera": "camera",
        "frame_sharding": "shard",
        "vision_models": "eye",
        "kimi_k2": "brain",
        "tts": "voice",
        "speaker": "speaker",
    }
    return [
        {"key": key, "name": labels[key], "icon": icons[key]}
        for key in PIPELINE_STAGES
    ]


def pipeline_subtitle() -> str:
    parts = [s["name"] for s in pipeline_stages()]
    return " → ".join(parts)


def status_chips() -> list[dict[str, str]]:
    return [
        {"key": "camera", "label": "Camera", "on": "Live"},
        {"key": "vision_models", "label": vision_model_label(), "on": "Running"},
        {"key": "kimi_k2", "label": reasoning_model_label(), "on": "Reasoning"},
        {"key": "tts", "label": "Voice", "on": "Streaming"},
    ]


def frontend_config() -> dict[str, object]:
    return {
        "pipeline": {
            "stages": pipeline_stages(),
            "subtitle": pipeline_subtitle(),
            "chips": status_chips(),
        },
        "vision_model": _vision_model(),
        "vision_model_label": vision_model_label(),
        "reasoning_model": _reasoning_model(),
        "reasoning_model_label": reasoning_model_label(),
        "tts_model": _tts_model(),
        "idle_guidance": "Monitoring your environment. I'll speak up if I detect danger.",
        "idle_situation": "Monitoring environment…",
    }
