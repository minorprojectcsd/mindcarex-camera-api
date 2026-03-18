"""
combined.py — Merges svc1 voice chunks + svc3 camera frames into a unified
              per-timestamp combined analysis.

Combined stress score formula:
  Both available : 0.55 × voice_stress + 0.45 × face_stress
  Voice only     : voice_stress
  Face only      : face_stress

Risk escalation rule:
  If voice risk == HIGH and face risk == HIGH → combined risk = HIGH
  If both express the SAME high-risk emotion (e.g. both fearful) → escalate to HIGH
  Otherwise → take the worse of the two
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.config  import VOICE_WEIGHT, FACE_WEIGHT
from app.models  import CameraSession, CameraFrame, VoiceChunkRO, VoiceSessionRO
from app.frame_analyzer import STATE_LABEL, STATE_COLOR


# ── State from combined score ─────────────────────────────────────────────────

def _combined_state(score: float) -> tuple[str, str, str]:
    """Returns (mental_state, label, color)."""
    if   score >= 72: state = "high_stress"
    elif score >= 50: state = "moderate_stress"
    elif score >= 30: state = "mild_stress"
    else:             state = "calm"
    return state, STATE_LABEL[state], STATE_COLOR[state]


def _combined_risk(
    voice_risk: str | None,
    face_risk: str | None,
    voice_emotions: list[dict],
    face_expressions: list[dict],
) -> str:
    """Compute combined risk — escalates if both modalities agree on danger."""
    high_risk_labels = {"fearful", "fearful"}

    # Both high → HIGH
    if voice_risk == "high" and face_risk == "high":
        return "high"

    # Voice high-risk emotion + matching face expression → escalate
    v_labels = {e.get("label", "") for e in voice_emotions if e.get("score", 0) > 0.30}
    f_labels = {e.get("label", "") for e in face_expressions if e.get("score", 0) > 0.30}
    if v_labels & f_labels & {"fearful", "angry", "disgusted"}:
        return "high"

    # Take worst
    order = {"high": 3, "medium": 2, "low": 1}
    worst = max(
        order.get(voice_risk or "low", 1),
        order.get(face_risk  or "low", 1),
    )
    return {3: "high", 2: "medium", 1: "low"}[worst]


# ── Time alignment helper ─────────────────────────────────────────────────────

def _nearest(items: list, timestamp: float, key: str = "timestamp_sec", tol: float = 8.0):
    """Find the item nearest in time within tolerance seconds."""
    best, best_diff = None, float("inf")
    for item in items:
        diff = abs((getattr(item, key, None) or item.get(key, 0)) - timestamp)
        if diff < best_diff and diff <= tol:
            best, best_diff = item, diff
    return best


# ── Main combined analysis builder ────────────────────────────────────────────

def build_combined(
    db: Session,
    camera_session: CameraSession,
) -> dict:
    """
    Align camera frames with voice chunks by timestamp.
    Returns combined timeline + summary.
    """
    voice_session_id = camera_session.voice_session_id
    frames: list[CameraFrame] = camera_session.frames

    # Pull voice chunks if linked
    voice_chunks: list[VoiceChunkRO] = []
    voice_summary = {}
    if voice_session_id:
        voice_chunks = (
            db.query(VoiceChunkRO)
            .filter(VoiceChunkRO.session_id == voice_session_id)
            .order_by(VoiceChunkRO.timestamp_sec)
            .all()
        )
        vs = db.query(VoiceSessionRO).filter(VoiceSessionRO.id == voice_session_id).first()
        if vs:
            voice_summary = vs.summary_json or {}

    # Build combined timeline — use ALL camera frames as anchor
    combined_timeline = []

    for frame in frames:
        if not frame.face_detected or frame.face_stress_score is None:
            continue

        # Find nearest voice chunk within 8s
        vc = _nearest(voice_chunks, frame.timestamp_sec)

        voice_stress    = vc.stress_score       if vc else None
        voice_risk      = vc.risk_level         if vc else None
        voice_emotions  = (vc.top_emotions_json or []) if vc else []
        voice_state     = vc.mental_state        if vc else None
        transcript      = vc.chunk_transcript    if vc else None

        face_stress = frame.face_stress_score
        face_risk   = frame.risk_level
        face_exprs  = frame.expressions_json or []

        # Combined stress
        if voice_stress is not None:
            combined = round(VOICE_WEIGHT * voice_stress + FACE_WEIGHT * face_stress, 1)
        else:
            combined = face_stress

        combined = min(max(combined, 0.0), 100.0)
        c_state, c_label, c_color = _combined_state(combined)
        c_risk = _combined_risk(voice_risk, face_risk, voice_emotions, face_exprs)

        combined_timeline.append({
            "timestamp_sec":       frame.timestamp_sec,
            "frame_index":         frame.frame_index,
            "voice_stress":        voice_stress,
            "face_stress":         face_stress,
            "combined_stress":     combined,
            "voice_mental_state":  voice_state,
            "face_mental_state":   frame.mental_state,
            "combined_state":      c_state,
            "combined_label":      c_label,
            "combined_color":      c_color,
            "combined_risk":       c_risk,
            "dominant_expression": frame.dominant_expression,
            "top_voice_emotions":  voice_emotions[:3],
            "top_face_expressions": sorted(face_exprs, key=lambda x: x.get("score", 0), reverse=True)[:3],
            "transcript_snippet":  transcript,
        })

    if not combined_timeline:
        return {
            "combined_timeline": [],
            "combined_summary":  {"error": "No aligned voice+face data"},
            "voice_summary":     voice_summary,
            "face_summary":      camera_session.summary_json or {},
        }

    # Combined summary stats
    c_scores  = [c["combined_stress"] for c in combined_timeline]
    avg_c     = round(sum(c_scores) / len(c_scores), 1)
    peak_c    = round(max(c_scores), 1)

    risk_levels  = [c["combined_risk"] for c in combined_timeline]
    overall_risk = "high" if "high" in risk_levels else ("medium" if "medium" in risk_levels else "low")

    states: dict[str, int] = {}
    for c in combined_timeline:
        states[c["combined_state"]] = states.get(c["combined_state"], 0) + 1
    dominant_state = max(states, key=states.get) if states else "calm"

    combined_summary = {
        "avg_combined_stress":  avg_c,
        "peak_combined_stress": peak_c,
        "overall_risk_level":   overall_risk,
        "dominant_state":       dominant_state,
        "dominant_label":       STATE_LABEL.get(dominant_state, dominant_state),
        "state_distribution":   states,
        "aligned_points":       len(combined_timeline),
        "voice_weight_used":    VOICE_WEIGHT,
        "face_weight_used":     FACE_WEIGHT,
    }

    return {
        "combined_timeline": combined_timeline,
        "combined_summary":  combined_summary,
        "voice_summary":     voice_summary,
        "face_summary":      camera_session.summary_json or {},
    }
