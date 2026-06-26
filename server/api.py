"""DisOps FastAPI application — REST control plane + WebSocket live feed + static UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import frontend_config
from .orchestrator import SessionManager
from .scenarios import scenario_list

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"

app = FastAPI(title="DisOps", description="Real-time vision-to-voice disaster response agent", version="1.0.0")
session = SessionManager()


class ScenarioBody(BaseModel):
    scenario: str


class ToggleBody(BaseModel):
    enabled: bool


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {"ok": True, "app": "DisOps", "status": session.status()}


@app.get("/api/config")
async def get_config() -> JSONResponse:
    return JSONResponse(frontend_config())


@app.get("/api/state")
async def get_state() -> JSONResponse:
    return JSONResponse(session.snapshot())


@app.get("/api/scenarios")
async def get_scenarios() -> JSONResponse:
    return JSONResponse({"scenarios": scenario_list()})


@app.post("/api/start")
async def start() -> dict[str, object]:
    await session.start()
    return {"ok": True, "status": session.status()}


@app.post("/api/stop")
async def stop() -> dict[str, object]:
    await session.stop()
    return {"ok": True, "status": session.status()}


@app.post("/api/pause")
async def pause(body: ToggleBody) -> dict[str, object]:
    await session.pause(body.enabled)
    return {"ok": True, "status": session.status()}


@app.post("/api/mute")
async def mute(body: ToggleBody) -> dict[str, object]:
    await session.set_muted(body.enabled)
    return {"ok": True, "status": session.status()}


@app.post("/api/demo-mode")
async def demo_mode(body: ToggleBody) -> dict[str, object]:
    await session.set_demo_mode(body.enabled)
    return {"ok": True, "status": session.status()}


@app.post("/api/scenario")
async def set_scenario(body: ScenarioBody) -> dict[str, object]:
    await session.set_scenario(body.scenario)
    return {"ok": True, "status": session.status()}


@app.post("/api/snapshot")
async def snapshot() -> dict[str, object]:
    await session.snapshot_capture()
    return {"ok": True}


@app.post("/api/replay")
async def replay() -> dict[str, object]:
    await session.replay_last()
    return {"ok": True}


@app.post("/api/clear")
async def clear() -> dict[str, object]:
    await session.clear_session()
    return {"ok": True, "status": session.status()}


class FrameBody(BaseModel):
    frame: str  # data URL or raw base64 JPEG/PNG


@app.post("/api/frame")
async def ingest_frame(body: FrameBody) -> JSONResponse:
    if not session.running:
        return JSONResponse({"error": "session not running"}, status_code=409)
    if session.demo_mode:
        return JSONResponse({"error": "disable demo mode for live frames"}, status_code=409)
    try:
        event = await session.ingest_frame(body.frame)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    return JSONResponse(event.to_dict())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await session.connect(websocket)
    try:
        while True:
            # Keep the socket alive; the client may send pings/frames as JSON.
            await websocket.receive_text()
    except WebSocketDisconnect:
        session.disconnect(websocket)
    except Exception:  # noqa: BLE001
        session.disconnect(websocket)


# --- static UI (mounted last so /api and /ws take precedence) ----------------
if WEB_DIR.is_dir():
    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
