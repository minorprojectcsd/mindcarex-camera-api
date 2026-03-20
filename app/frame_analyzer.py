"""
frame_analyzer.py — Facial expression analysis via HuggingFace Inference API

Model: trpakov/vit-face-expression (ViT-B/16)
Free HuggingFace Inference API — no GPU needed
"""

from __future__ import annotations

import logging
import time
import requests

from app.config import HF_API_TOKEN

# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger("frame_analyzer")

HF_MODEL_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "trpakov/vit-face-expression"
)

# Expression → Stress weights
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
    "calm": "green",
    "mild_stress": "yellow",
    "moderate_stress": "orange",
    "high_stress": "red",
}

# ─────────────────────────────────────────────────────────────────────────────
# 🔹 HuggingFace API Call
# ─────────────────────────────────────────────────────────────────────────────

def detect_expressions(image_bytes: bytes) -> list[dict]:
    """
    Sends image to HuggingFace model and returns:
    [{label, score}, ...]
    """

    if not HF_API_TOKEN:
        log.warning("HF_API_TOKEN not set")
        return []

    headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "image/jpeg",
    }

    retry_delays = [15, 20, 25]

    try:
        for attempt in range(4):
            response = requests.post(
                HF_MODEL_URL,
                headers=headers,
                data=image_bytes,
                timeout=45,
            )

            # Handle cold start
            if response.status_code == 503 and attempt < 3:
                delay = retry_delays[attempt]
                log.info(f"HF cold start → retry {attempt+1} in {delay}s")
                time.sleep(delay)
                continue

            break

        if response.status_code != 200:
            log.warning(f"HF API Error {response.status_code}: {response.text[:120]}")
            return []

        data = response.json()

        # Normalize response
        if isinstance(data, list) and data:
            if isinstance(data[0], list):
                data = data[0]

            return [
                {
                    "label": item["label"].lower().strip(),
                    "score": round(float(item["score"]), 4),
                }
                for item in data
            ]

        return []

    except requests.Timeout:
        log.warning("HF API timeout")
        return []

    except Exception as e:
        log.warning(f"HF API failure: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 🔹 Stress Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_face_stress(expressions: list[dict]) -> dict:
    """
    Converts emotion scores → stress score (0–100)
    """

    if not expressions:
        return {}

    emotion_map = {e["label"]: e["score"] for e in expressions}

    raw_score = sum(
        emotion_map.get(expr, 0.0) * weight
        for expr, weight in EXPRESSION_STRESS_WEIGHT.items()
    )

    score = round(min(max(raw_score * 100, 0), 100), 1)

    # State classification
    if score >= 72:
        state = "high_stress"
    elif score >= 50:
        state = "moderate_stress"
    elif score >= 30:
        state = "mild_stress"
    else:
        state = "calm"

    dominant = max(expressions, key=lambda x: x["score"])["label"]

    detected = {e["label"] for e in expressions if e["score"] > 0.30}

    if detected & HIGH_RISK_EXPR:
        risk = "high"
    elif detected & MEDIUM_RISK_EXPR:
        risk = "medium"
    elif score >= 65:
        risk = "medium"
    else:
        risk = "low"

    return {
        "face_stress_score": score,
        "mental_state": state,
        "mental_state_label": STATE_LABEL[state],
        "color": STATE_COLOR[state],
        "risk_level": risk,
        "dominant_expression": dominant,
        "top_expressions": sorted(expressions, key=lambda x: x["score"], reverse=True)[:5],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 🔹 Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_frame(
    image_bytes: bytes,
    frame_index: int,
    timestamp_sec: float,
) -> dict:
    """
    End-to-end processing:
    frame → expressions → stress analysis
    """

    expressions = detect_expressions(image_bytes)

    if not expressions:
        return {
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "face_detected": False,
            "face_stress_score": None,
            "mental_state": None,
            "mental_state_label": None,
            "color": None,
            "risk_level": "low",
            "dominant_expression": None,
            "top_expressions": [],
            "mode": "no_token" if not HF_API_TOKEN else "api_unavailable",
        }

    stress = compute_face_stress(expressions)

    return {
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "face_detected": True,
        "face_stress_score": stress["face_stress_score"],
        "mental_state": stress["mental_state"],
        "mental_state_label": stress["mental_state_label"],
        "color": stress["color"],
        "risk_level": stress["risk_level"],
        "dominant_expression": stress["dominant_expression"],
        "top_expressions": stress["top_expressions"],
        "mode": "hf_api",
    }
