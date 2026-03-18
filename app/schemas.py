from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class StartSessionRequest(BaseModel):
    patient_id:       str = "unknown"
    label:            str = "Camera Session"
    voice_session_id: Optional[str] = None   # link to svc1 session


class StopSessionRequest(BaseModel):
    session_id: str


class ExpressionScore(BaseModel):
    label: str
    score: float


class FrameResponse(BaseModel):
    session_id:         str
    frame_index:        int
    timestamp_sec:      float
    face_detected:      bool
    face_stress_score:  Optional[float]
    mental_state:       Optional[str]
    mental_state_label: Optional[str]
    color:              Optional[str]
    risk_level:         str
    dominant_expression: Optional[str]
    expressions:        list[ExpressionScore]
    mode:               str
    total_frames:       int


class CombinedChunk(BaseModel):
    """One time-aligned combined voice + face data point."""
    timestamp_sec:       float
    voice_stress:        Optional[float]
    face_stress:         Optional[float]
    combined_stress:     float
    voice_mental_state:  Optional[str]
    face_mental_state:   Optional[str]
    combined_state:      str
    combined_label:      str
    combined_color:      str
    combined_risk:       str
    dominant_expression: Optional[str]
    top_voice_emotions:  list[dict]
    transcript_snippet:  Optional[str]
