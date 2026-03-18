"""
frame_analyzer.py — Facial expression analysis via HuggingFace Inference API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model used:  trpakov/vit-face-expression
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Architecture : Vision Transformer (ViT-B/16)
  Trained on   : FER2013 + AffectNet datasets
  Labels       : angry · disgusted · fearful · happy · neutral · sad · surprised
  Hosted by    : HuggingFace (FREE Inference API)
  Input        : JPEG/PNG image bytes (one frame from the video call)
  Output       : [{label, score}, ...] sorted by score desc
  Your server  : sends ~50 KB JPEG via HTTP — model runs on HF servers
  No GPU/RAM   : zero heavy dependencies on your end
  Same token   : HF_API_TOKEN from svc1 — no extra key needed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Face stress score (0-100):
  Weighted sum of expression scores mapped to stress axis:
    fearful   → 1.00   (max stress)
    angry     → 0.85
    disgusted → 0.70
    sad       → 0.55
    surprised → 0.30
    neutral   → 0.05
    happy     → -0.20  (reduces stress score)

Mental state thresholds (same as svc1 for consistency):
  72-100 → high_stress   (red)
  50-71  → moderate_stress (orange)
  30-49  → mild_stress   (yellow)
  0-29   → calm          (green)
"""
from __future__ import annotations

import logging
import time
import io

import requests

from app.config import HF_API_TOKEN

log = logging.getLogger("frame_analyzer")

HF_MODEL_URL = (
    "https://api-inference.huggingface.co/models/"
    "trpakov/vit-face-expression"
)

EXPRESSION_STRESS_WEIGHT: dict[str, float] = {
    "fearful":   1.00,
    "angry":     0.85,
    "disgusted": 0.70,
    "sad":       0.55,
    "surprised": 0.30,
    "neutral":   0.05,
    "happy":    -0.20,
}

HIGH_RISK_EXPR   = {"fearful"}
MEDIUM_RISK_EXPR = {"angry", "disgusted", "sad"}

STATE_LABEL = {
    "calm":            "Calm / Relaxed",
    "mild_stress":     "Mild Stress",
    "moderate_stress": "Moderate Stress",
    "high_stress":     "High Stress",
}
STATE_COLOR = {
    "calm": "green", "mild_stress": "yellow",
    "moderate_stress": "orange", "high_stress": "red",
}


# ── HuggingFace Inference API call ────────────────────────────────────────────

def detect_expressions(image_bytes: bytes) -> list[dict]:
    """
    Send JPEG/PNG frame bytes to HuggingFace ViT face expression model.
    Returns [{label, score}, ...] sorted by score desc.
    Returns [] if token not set or call fails.

    Note: First call may take ~20s (HF cold-starts the model).
    Subsequent calls: ~1-3s.
    """
    if not HF_API_TOKEN:
        return []

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    try:
        resp = requests.post(
            HF_MODEL_URL,
            headers=headers,
            data=image_bytes,
            timeout=30,
        )

        if resp.status_code == 503:
            log.info("HF face model cold-starting — retrying in 15s")
            time.sleep(15)
            resp = requests.post(
                HF_MODEL_URL,
                headers=headers,
                data=image_bytes,
                timeout=30,
            )

        if resp.status_code != 200:
            log.warning(f"HF face API {resp.status_code}: {resp.text[:100]}")
            return []

        results = resp.json()

        # HF returns either a list or a list-of-lists depending on model version
        if isinstance(results, list) and len(results) > 0:
            # Flatten if nested
            if isinstance(results[0], list):
                results = results[0]
            return [
                {
                    "label": r["label"].lower().strip(),
                    "score": round(float(r["score"]), 4),
                }
                for r in results
            ]

        return []

    except requests.Timeout:
        log.warning("HF face API timeout — skipping frame")
        return []
    except Exception as e:
        log.warning(f"HF face API error: {e}")
        return []


# ── Face stress score ─────────────────────────────────────────────────────────

def compute_face_stress(expressions: list[dict]) -> dict:
    """
    Convert expression scores into a 0-100 face stress score.
    Returns empty dict if no expressions (no face detected).
    """
    if not expressions:
        return {}

    emo_map = {e["label"]: e["score"] for e in expressions}

    # Weighted stress score
    raw = sum(
        emo_map.get(expr, 0.0) * weight
        for expr, weight in EXPRESSION_STRESS_WEIGHT.items()
    )
    score = round(min(max(raw * 100, 0.0), 100.0), 1)

    # Mental state
    if   score >= 72: state = "high_stress"
    elif score >= 50: state = "moderate_stress"
    elif score >= 30: state = "mild_stress"
    else:             state = "calm"

    # Dominant expression
    dominant = max(expressions, key=lambda x: x["score"])["label"]

    # Risk level
    detected = {e["label"] for e in expressions if e["score"] > 0.30}
    if   detected & HIGH_RISK_EXPR:   risk = "high"
    elif detected & MEDIUM_RISK_EXPR: risk = "medium"
    elif score >= 65:                  risk = "medium"
    else:                              risk = "low"

    return {
        "face_stress_score":   score,
        "mental_state":        state,
        "mental_state_label":  STATE_LABEL[state],
        "color":               STATE_COLOR[state],
        "risk_level":          risk,
        "dominant_expression": dominant,
        "top_expressions":     sorted(expressions, key=lambda x: x["score"], reverse=True)[:5],
    }


# ── Full frame pipeline ───────────────────────────────────────────────────────

def process_frame(
    image_bytes: bytes,
    frame_index: int,
    timestamp_sec: float,
) -> dict:
    """
    Full pipeline: image bytes → HF API → expression scores → stress result.
    Returns a dict ready to be stored as a CameraFrame record.
    """
    expressions = detect_expressions(image_bytes)

    if not expressions:
        # No face detected or API unavailable
        return {
            "frame_index":        frame_index,
            "timestamp_sec":      timestamp_sec,
            "face_detected":      False,
            "face_stress_score":  None,
            "mental_state":       None,
            "mental_state_label": None,
            "color":              None,
            "risk_level":         "low",
            "dominant_expression": None,
            "top_expressions":    [],
            "mode":               "no_face" if not HF_API_TOKEN else "api_unavailable",
        }

    stress = compute_face_stress(expressions)

    return {
        "frame_index":        frame_index,
        "timestamp_sec":      timestamp_sec,
        "face_detected":      True,
        "face_stress_score":  stress["face_stress_score"],
        "mental_state":       stress["mental_state"],
        "mental_state_label": stress["mental_state_label"],
        "color":              stress["color"],
        "risk_level":         stress["risk_level"],
        "dominant_expression": stress["dominant_expression"],
        "top_expressions":    stress["top_expressions"],
        "mode":               "hf_api",
    }
