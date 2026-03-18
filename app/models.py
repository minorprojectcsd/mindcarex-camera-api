from __future__ import annotations
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    Text, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from app.database import Base


class CameraSession(Base):
    __tablename__ = "camera_sessions"

    id               = Column(String,   primary_key=True)
    patient_id       = Column(String,   nullable=False, index=True)
    label            = Column(String,   default="Camera Session")
    # Links to svc1 voice session — null if camera-only
    voice_session_id = Column(String,   nullable=True, index=True)
    status           = Column(String,   default="recording")   # recording | completed
    started_at       = Column(DateTime, default=datetime.utcnow)
    ended_at         = Column(DateTime, nullable=True)
    summary_json     = Column(JSON,     nullable=True)

    frames = relationship(
        "CameraFrame",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="CameraFrame.frame_index",
        lazy="select",
    )

    def to_dict(self) -> dict:
        return {
            "session_id":       self.id,
            "patient_id":       self.patient_id,
            "label":            self.label,
            "voice_session_id": self.voice_session_id,
            "status":           self.status,
            "started_at":       self.started_at.isoformat() if self.started_at else None,
            "ended_at":         self.ended_at.isoformat()   if self.ended_at   else None,
            "summary":          self.summary_json,
            "frame_count":      len(self.frames) if self.frames else 0,
        }


class CameraFrame(Base):
    __tablename__ = "camera_frames"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    session_id          = Column(String,  ForeignKey("camera_sessions.id"), nullable=False, index=True)
    frame_index         = Column(Integer, nullable=False)
    timestamp_sec       = Column(Float,   default=0.0)
    processed_at        = Column(DateTime, default=datetime.utcnow)

    # Did the model detect a face in this frame?
    face_detected       = Column(Boolean, default=False)

    # Face stress score 0-100 (null if no face detected)
    face_stress_score   = Column(Float,  nullable=True)
    mental_state        = Column(String, nullable=True)
    mental_state_label  = Column(String, nullable=True)
    color               = Column(String, nullable=True)
    risk_level          = Column(String, default="low")

    # Dominant single expression (e.g. "fearful")
    dominant_expression = Column(String, nullable=True)

    # Full expression scores: [{label, score}, ...]
    expressions_json    = Column(JSON, nullable=True)

    # mode: "hf_api" | "no_face" | "api_unavailable"
    mode                = Column(String, default="hf_api")

    session = relationship("CameraSession", back_populates="frames")

    def to_dict(self) -> dict:
        return {
            "frame_index":        self.frame_index,
            "timestamp_sec":      self.timestamp_sec,
            "face_detected":      self.face_detected,
            "face_stress_score":  self.face_stress_score,
            "mental_state":       self.mental_state,
            "mental_state_label": self.mental_state_label,
            "color":              self.color,
            "risk_level":         self.risk_level,
            "dominant_expression": self.dominant_expression,
            "expressions":        self.expressions_json or [],
            "mode":               self.mode,
            "processed_at":       self.processed_at.isoformat() if self.processed_at else None,
        }


# ── Read-only mirrors of svc1 tables (for combined score endpoint) ─────────────

class VoiceChunkRO(Base):
    """Read-only mirror of svc1 voice_chunks — used in combined analysis."""
    __tablename__ = "voice_chunks"

    id                 = Column(Integer, primary_key=True)
    session_id         = Column(String,  index=True)
    chunk_index        = Column(Integer)
    timestamp_sec      = Column(Float)
    stress_score       = Column(Float)
    mental_state       = Column(String)
    mental_state_label = Column(String)
    color              = Column(String)
    risk_level         = Column(String)
    top_emotions_json  = Column(JSON)
    chunk_transcript   = Column(Text)


class VoiceSessionRO(Base):
    """Read-only mirror of svc1 voice_sessions — for combined summary."""
    __tablename__ = "voice_sessions"

    id              = Column(String, primary_key=True)
    patient_id      = Column(String)
    label           = Column(String)
    status          = Column(String)
    full_transcript = Column(Text)
    summary_json    = Column(JSON)
