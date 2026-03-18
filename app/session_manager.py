"""
session_manager.py — All Neon DB operations for camera sessions.
"""
from __future__ import annotations
import uuid
from collections import defaultdict
from datetime import datetime

import numpy as np
from sqlalchemy.orm import Session

from app.models import CameraSession, CameraFrame
from app.frame_analyzer import STATE_LABEL


def create_session(
    db: Session,
    patient_id: str,
    label: str,
    voice_session_id: str | None,
) -> CameraSession:
    s = CameraSession(
        id=str(uuid.uuid4()),
        patient_id=patient_id,
        label=label,
        voice_session_id=voice_session_id,
        status="recording",
        started_at=datetime.utcnow(),
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def get_session(db: Session, session_id: str) -> CameraSession | None:
    return db.query(CameraSession).filter(CameraSession.id == session_id).first()


def get_patient_sessions(db: Session, patient_id: str) -> list[CameraSession]:
    return (
        db.query(CameraSession)
        .filter(CameraSession.patient_id == patient_id)
        .order_by(CameraSession.started_at.desc())
        .all()
    )


def append_frame(db: Session, session: CameraSession, data: dict) -> CameraFrame:
    frame = CameraFrame(
        session_id=session.id,
        frame_index=data["frame_index"],
        timestamp_sec=data["timestamp_sec"],
        face_detected=data["face_detected"],
        face_stress_score=data.get("face_stress_score"),
        mental_state=data.get("mental_state"),
        mental_state_label=data.get("mental_state_label"),
        color=data.get("color"),
        risk_level=data.get("risk_level", "low"),
        dominant_expression=data.get("dominant_expression"),
        expressions_json=data.get("top_expressions", []),
        mode=data.get("mode", "hf_api"),
        processed_at=datetime.utcnow(),
    )
    db.add(frame)
    db.commit()
    db.refresh(frame)
    return frame


def finalise_session(db: Session, session: CameraSession) -> dict:
    """Compute aggregate summary from all frames, store in Neon, mark completed."""
    frames: list[CameraFrame] = (
        db.query(CameraFrame)
        .filter(CameraFrame.session_id == session.id)
        .order_by(CameraFrame.frame_index)
        .all()
    )

    if not frames:
        summary = {"error": "No frames recorded"}
        session.status   = "completed"
        session.ended_at = datetime.utcnow()
        session.summary_json = summary
        db.commit()
        return summary

    # Only consider frames where a face was detected
    detected = [f for f in frames if f.face_detected and f.face_stress_score is not None]
    total    = len(frames)
    detected_count = len(detected)
    detection_rate = round(detected_count / total, 3) if total > 0 else 0.0

    if not detected:
        summary = {
            "error":          "No faces detected across session",
            "total_frames":   total,
            "detection_rate": 0.0,
        }
        session.status       = "completed"
        session.ended_at     = datetime.utcnow()
        session.summary_json = summary
        db.commit()
        return summary

    scores = [f.face_stress_score for f in detected]
    avg    = round(sum(scores) / len(scores), 1)
    peak   = round(max(scores), 1)
    low    = round(min(scores), 1)

    # Trend
    if len(scores) >= 3:
        x     = np.arange(len(scores), dtype=float)
        slope = float(np.polyfit(x, scores, 1)[0])
        trend = "worsening" if slope > 1.5 else ("improving" if slope < -1.5 else "stable")
    else:
        slope, trend = 0.0, "insufficient_data"

    # State distribution
    state_counts: dict[str, int] = defaultdict(int)
    for f in detected:
        state_counts[f.mental_state or "calm"] += 1
    dominant = max(state_counts, key=state_counts.get)

    # Risk
    risks        = [f.risk_level for f in detected]
    overall_risk = "high" if "high" in risks else ("medium" if "medium" in risks else "low")

    # Expression averages
    expr_totals: dict[str, float] = defaultdict(float)
    expr_counts: dict[str, int]   = defaultdict(int)
    for f in detected:
        for e in (f.expressions_json or []):
            expr_totals[e["label"]] += e["score"]
            expr_counts[e["label"]] += 1
    top_expressions = sorted(
        [
            {"label": k, "avg_score": round(expr_totals[k] / expr_counts[k], 4)}
            for k in expr_totals
        ],
        key=lambda x: x["avg_score"], reverse=True,
    )[:7]

    # Dominant expression across session
    expr_dom: dict[str, int] = defaultdict(int)
    for f in detected:
        if f.dominant_expression:
            expr_dom[f.dominant_expression] += 1
    session_dominant_expr = max(expr_dom, key=expr_dom.get) if expr_dom else None

    summary = {
        "avg_face_stress":         avg,
        "peak_face_stress":        peak,
        "min_face_stress":         low,
        "trend":                   trend,
        "trend_slope":             round(slope, 3),
        "dominant_mental_state":   dominant,
        "dominant_label":          STATE_LABEL.get(dominant, dominant),
        "overall_risk_level":      overall_risk,
        "total_frames":            total,
        "detected_frames":         detected_count,
        "detection_rate":          detection_rate,
        "session_dominant_expression": session_dominant_expr,
        "state_distribution":      dict(state_counts),
        "top_expressions":         top_expressions,
    }

    session.status       = "completed"
    session.ended_at     = datetime.utcnow()
    session.summary_json = summary
    db.commit()
    return summary
