"""
router.py — All SVC 3 endpoints

REST
  GET  /health
  GET  /status
  POST /session/start
  POST /{session_id}/frame
  POST /session/stop
  GET  /{session_id}
  GET  /{session_id}/timeline
  GET  /{session_id}/summary
  GET  /{session_id}/combined     ← voice + face merged
  GET  /patient/{patient_id}/history

WebSocket
  WS   /{session_id}/live         ← pushes each frame result in real-time
"""
from __future__ import annotations
import asyncio
import logging
from typing import Annotated

from fastapi import (
    APIRouter, Depends, File, Form,
    HTTPException, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database       import get_db
from app.models         import CameraSession
from app.schemas        import StartSessionRequest, StopSessionRequest
from app                import frame_analyzer as fa
from app                import session_manager as mgr
from app                import combined as comb

router = APIRouter()
log    = logging.getLogger("camera_router")


# ── WebSocket manager ─────────────────────────────────────────────────────────

class _WS:
    def __init__(self):
        self._c: dict[str, list[WebSocket]] = {}

    async def connect(self, sid: str, ws: WebSocket):
        await ws.accept()
        self._c.setdefault(sid, []).append(ws)

    def drop(self, sid: str, ws: WebSocket):
        if ws in self._c.get(sid, []):
            self._c[sid].remove(ws)

    async def push(self, sid: str, data: dict):
        for ws in list(self._c.get(sid, [])):
            try:
                await ws.send_json(data)
            except Exception:
                self.drop(sid, ws)


_ws = _WS()


# ── Helpers ───────────────────────────────────────────────────────────────────

def ok(data: dict, code: int = 200) -> JSONResponse:
    return JSONResponse({"success": True,  "data":  data}, status_code=code)

def err(msg: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"success": False, "error": msg},  status_code=code)

def get_or_404(db: Session, sid: str) -> CameraSession:
    s = mgr.get_session(db, sid)
    if not s:
        raise HTTPException(404, f"Camera session {sid} not found")
    return s


# ══════════════════════════════════════════════════════════════════════════════
# GET /health  |  GET /status
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/status")
def status():
    return ok({
        "service":        "svc3_camera_analysis",
        "hf_face_model":  bool(fa.HF_API_TOKEN),
        "model":          "trpakov/vit-face-expression",
        "note":           "Works without HF_API_TOKEN but no expression scores",
    })


# ══════════════════════════════════════════════════════════════════════════════
# POST /session/start
# Body: { patient_id, label, voice_session_id (optional) }
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/session/start", status_code=201)
def start_session(
    body: StartSessionRequest,
    db:   Annotated[Session, Depends(get_db)],
):
    s = mgr.create_session(db, body.patient_id, body.label, body.voice_session_id)
    return ok({
        "session_id":       s.id,
        "patient_id":       body.patient_id,
        "label":            body.label,
        "voice_session_id": body.voice_session_id,
        "status":           "recording",
        "hint":             f"POST frames every 5-10s to /{s.id}/frame",
    }, 201)


# ══════════════════════════════════════════════════════════════════════════════
# POST /{session_id}/frame
# form-data: file=<JPEG/PNG>  frame_index=-1  timestamp_sec=-1
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/{session_id}/frame")
async def upload_frame(
    session_id:    str,
    db:            Annotated[Session, Depends(get_db)],
    file:          UploadFile = File(...),
    frame_index:   int   = Form(default=-1),
    timestamp_sec: float = Form(default=-1.0),
):
    """
    Upload one video frame (JPEG/PNG).
    Sends to HuggingFace ViT face expression model.
    Returns face stress score + top expressions instantly.
    Call every 5-10 seconds during video session.
    """
    s           = get_or_404(db, session_id)
    image_bytes = await file.read()

    if s.status == "completed":
        return err("Session already completed")
    if len(image_bytes) < 100:
        return err("Image too small (< 100 bytes)")

    existing = len(s.frames)
    if frame_index < 0:
        frame_index = existing
    if timestamp_sec < 0:
        timestamp_sec = existing * 7.0   # assume 7s per frame

    # Run in thread pool (non-blocking)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, fa.process_frame, image_bytes, frame_index, timestamp_sec
        )
    except Exception as e:
        log.error(f"Frame analysis error: {e}")
        return err(f"Frame analysis failed: {e}", 500)

    frame = mgr.append_frame(db, s, result)

    response = {
        "session_id":          session_id,
        "frame_index":         frame.frame_index,
        "timestamp_sec":       frame.timestamp_sec,
        "face_detected":       frame.face_detected,
        "face_stress_score":   frame.face_stress_score,
        "mental_state":        frame.mental_state,
        "mental_state_label":  frame.mental_state_label,
        "color":               frame.color,
        "risk_level":          frame.risk_level,
        "dominant_expression": frame.dominant_expression,
        "expressions":         frame.expressions_json or [],
        "mode":                frame.mode,
        "total_frames":        existing + 1,
    }

    # Broadcast to WebSocket clients
    await _ws.push(session_id, {"event": "frame_result", **response})

    return ok(response)


# ══════════════════════════════════════════════════════════════════════════════
# POST /session/stop
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/session/stop")
def stop_session(
    body: StopSessionRequest,
    db:   Annotated[Session, Depends(get_db)],
):
    s       = get_or_404(db, body.session_id)
    summary = mgr.finalise_session(db, s)
    return ok({
        "session_id": body.session_id,
        "status":     "completed",
        "summary":    summary,
    })


# ══════════════════════════════════════════════════════════════════════════════
# WS /{session_id}/live
# ══════════════════════════════════════════════════════════════════════════════

@router.websocket("/{session_id}/live")
async def ws_live(session_id: str, ws: WebSocket):
    """
    Receives each frame result within ~100ms of processing.
    Message: { event, session_id, frame_index, face_detected,
               face_stress_score, mental_state, color, dominant_expression, ... }
    """
    await _ws.connect(session_id, ws)
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        _ws.drop(session_id, ws)


# ══════════════════════════════════════════════════════════════════════════════
# GET /{session_id}  — full session + all frames
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/{session_id}")
def get_session(session_id: str, db: Annotated[Session, Depends(get_db)]):
    s = get_or_404(db, session_id)
    return ok({**s.to_dict(), "frames": [f.to_dict() for f in s.frames]})


# ══════════════════════════════════════════════════════════════════════════════
# GET /{session_id}/timeline  — face stress per frame (for chart)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/{session_id}/timeline")
def get_timeline(session_id: str, db: Annotated[Session, Depends(get_db)]):
    s = get_or_404(db, session_id)
    timeline = [
        {
            "frame_index":         f.frame_index,
            "timestamp_sec":       f.timestamp_sec,
            "face_detected":       f.face_detected,
            "face_stress_score":   f.face_stress_score,
            "mental_state":        f.mental_state,
            "label":               f.mental_state_label,
            "color":               f.color,
            "risk_level":          f.risk_level,
            "dominant_expression": f.dominant_expression,
        }
        for f in s.frames
    ]
    return ok({"session_id": session_id, "count": len(timeline), "timeline": timeline})


# ══════════════════════════════════════════════════════════════════════════════
# GET /{session_id}/summary
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/{session_id}/summary")
def get_summary(session_id: str, db: Annotated[Session, Depends(get_db)]):
    s = get_or_404(db, session_id)
    return ok({
        "session_id":       session_id,
        "status":           s.status,
        "voice_session_id": s.voice_session_id,
        "summary":          s.summary_json or {"note": "Session still recording"},
    })


# ══════════════════════════════════════════════════════════════════════════════
# GET /{session_id}/combined  — merged voice + face per timestamp
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/{session_id}/combined")
def get_combined(session_id: str, db: Annotated[Session, Depends(get_db)]):
    """
    Time-aligns camera frames with svc1 voice chunks.
    Returns unified combined_stress score per point + combined summary.
    Requires voice_session_id to have been set when session was started.
    """
    s = get_or_404(db, session_id)
    result = comb.build_combined(db, s)
    return ok({"session_id": session_id, **result})


# ══════════════════════════════════════════════════════════════════════════════
# GET /patient/{patient_id}/history
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/patient/{patient_id}/history")
def get_history(patient_id: str, db: Annotated[Session, Depends(get_db)]):
    sessions = mgr.get_patient_sessions(db, patient_id)
    return ok({
        "patient_id":     patient_id,
        "total_sessions": len(sessions),
        "sessions": [
            {
                "session_id":       s.id,
                "label":            s.label,
                "status":           s.status,
                "voice_session_id": s.voice_session_id,
                "started_at":       s.started_at.isoformat() if s.started_at else None,
                "ended_at":         s.ended_at.isoformat()   if s.ended_at   else None,
                "frame_count":      len(s.frames),
                "summary":          s.summary_json,
            }
            for s in sessions
        ],
    })
