# 📷 MindCareX — Camera Analysis Service (SVC3)

> Real-time facial expression analysis from video frames during consultations.
> Analyses faces every 5–10 seconds and combines with voice data for unified mental state scoring.

---

## What This Service Does

During a video consultation, the frontend captures a JPEG frame from the patient's camera every 5–10 seconds and sends it here. This service:

1. Sends the image to HuggingFace `trpakov/vit-face-expression` model
2. Gets back 7 expression scores (angry, disgusted, fearful, happy, neutral, sad, surprised)
3. Computes a face stress score 0–100
4. Saves the frame result to Neon
5. Broadcasts the result to the doctor's screen via WebSocket
6. On session end, computes a full face analysis summary
7. Optionally combines with SVC1 voice data for a unified stress score

---

## Face Stress Score Formula

```
face_stress = sum(expression_score × weight) × 100

Weights:
  fearful   × 1.00   (max stress)
  angry     × 0.85
  disgusted × 0.70
  sad       × 0.55
  surprised × 0.30
  neutral   × 0.05
  happy     × -0.20  (reduces stress)
```

| Score | State | Color |
|-------|-------|-------|
| 72–100 | High Stress | 🔴 Red |
| 50–71 | Moderate Stress | 🟠 Orange |
| 30–49 | Mild Stress | 🟡 Yellow |
| 0–29 | Calm / Relaxed | 🟢 Green |

---

## Combined Score (Voice + Face)

When linked to a SVC1 voice session:

```
combined_stress = 0.55 × voice_stress + 0.45 × face_stress
```

**Risk escalation rule:** If voice risk AND face risk are both HIGH, or if both modalities show the same high-risk emotion > 30%, combined risk automatically escalates to HIGH.

Weights are configurable via `VOICE_WEIGHT` and `FACE_WEIGHT` env vars.

---

## File Structure

```
svc3/
├── main.py                  FastAPI app, creates DB tables on startup
├── requirements.txt
├── Dockerfile               python:3.11-slim + Pillow
├── .env.example
└── app/
    ├── config.py            env vars + weight configuration
    ├── database.py          Neon SQLAlchemy engine
    ├── models.py            CameraSession + CameraFrame tables
    │                        + VoiceChunkRO (read-only mirror of SVC1)
    ├── schemas.py           Pydantic request/response models
    ├── frame_analyzer.py    HuggingFace ViT face model + stress computation
    ├── session_manager.py   all DB ops + aggregate face summary
    ├── combined.py          time-aligns voice chunks + face frames
    └── router.py            all REST endpoints + WebSocket manager
```

---

## Database Tables

Creates and owns:

**`camera_sessions`**
```
id, patient_id, label, voice_session_id (FK to SVC1),
status, started_at, ended_at, summary_json
```

**`camera_frames`**
```
id, session_id, frame_index, timestamp_sec,
face_detected, face_stress_score, mental_state, mental_state_label,
color, risk_level, dominant_expression, expressions_json, mode, processed_at
```

Reads (read-only):
- `voice_chunks` — for combined score alignment by timestamp

---

## AI Model

| Detail | Value |
|--------|-------|
| Model | `trpakov/vit-face-expression` |
| Architecture | Vision Transformer (ViT-B/16) |
| Training data | FER2013 + AffectNet |
| Provider | HuggingFace Inference API (free) |
| API URL | `https://router.huggingface.co/hf-inference/models/trpakov/vit-face-expression` |
| Input | JPEG/PNG image bytes |
| Output | 7 expression scores |
| Token needed | HF token with "Make calls to Inference Providers" permission |

**⚠️ Important (March 2026):** HuggingFace deprecated `api-inference.huggingface.co`. All calls must use `router.huggingface.co/hf-inference/models/`. The `HF_API_TOKEN` must be a **Fine-grained** token with "Make calls to serverless Inference Providers" enabled.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `GET` | `/api/camera/status` | Shows if HF face model token is loaded |
| `POST` | `/api/camera/session/start` | Start session. Body: `{patient_id, label, voice_session_id}` |
| `POST` | `/api/camera/{id}/frame` | Upload JPEG frame (multipart: file). Returns face stress + expressions |
| `WS` | `/api/camera/{id}/live` | WebSocket — doctor screen receives frame result in real time |
| `POST` | `/api/camera/session/stop` | End session, compute face summary |
| `GET` | `/api/camera/{id}` | Full session + all frames |
| `GET` | `/api/camera/{id}/timeline` | Face stress per frame (for chart) |
| `GET` | `/api/camera/{id}/summary` | Aggregate: avg/peak face stress, detection rate, dominant expression |
| `GET` | `/api/camera/{id}/combined` | Time-aligned voice + face → unified combined stress score |
| `GET` | `/api/camera/patient/{id}/history` | All camera sessions for a patient |

### Frame upload response shape

```json
{
  "success": true,
  "data": {
    "session_id": "uuid",
    "frame_index": 9,
    "timestamp_sec": 63.0,
    "face_detected": true,
    "face_stress_score": 4.2,
    "mental_state": "calm",
    "mental_state_label": "Calm / Relaxed",
    "color": "green",
    "risk_level": "low",
    "dominant_expression": "happy",
    "expressions": [
      {"label": "happy",   "score": 0.8835},
      {"label": "neutral", "score": 0.1065},
      {"label": "surprise","score": 0.0051}
    ],
    "mode": "hf_api",
    "total_frames": 10
  }
}
```

### Combined endpoint response shape

```json
{
  "success": true,
  "data": {
    "session_id": "uuid",
    "combined_timeline": [
      {
        "timestamp_sec": 7.0,
        "voice_stress": 57.5,
        "face_stress": 4.2,
        "combined_stress": 33.5,
        "voice_mental_state": "moderate_stress",
        "face_mental_state": "calm",
        "combined_state": "mild_stress",
        "combined_label": "Mild Stress",
        "combined_color": "yellow",
        "combined_risk": "low",
        "dominant_expression": "happy",
        "transcript_snippet": "I am feeling better today"
      }
    ],
    "combined_summary": {
      "avg_combined_stress": 33.5,
      "peak_combined_stress": 45.2,
      "overall_risk_level": "low",
      "dominant_state": "mild_stress"
    },
    "voice_summary": {...},
    "face_summary": {...}
  }
}
```

### WebSocket

Connect to `ws://host/api/camera/{session_id}/live`:

```json
{
  "event": "frame_result",
  "session_id": "uuid",
  "face_detected": true,
  "face_stress_score": 4.2,
  "mental_state": "calm",
  "dominant_expression": "happy",
  "expressions": [...],
  "color": "green"
}
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ Yes | Same Neon connection string as SVC1 + SVC2 |
| `HF_API_TOKEN` | ✅ Yes | HuggingFace token with Inference Providers permission |
| `HF_MODEL_URL` | Optional | Default: `https://router.huggingface.co/hf-inference/models/trpakov/vit-face-expression` |
| `VOICE_WEIGHT` | Optional | Default: `0.55` — voice weight in combined score |
| `FACE_WEIGHT` | Optional | Default: `0.45` — face weight in combined score |
| `ALLOWED_ORIGINS` | Optional | CORS origins. Default: `http://localhost:5173` |

### `.env.example`

```env
DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require
HF_API_TOKEN=hf_xxxxxxxxxxxx
HF_MODEL_URL=https://router.huggingface.co/hf-inference/models/trpakov/vit-face-expression
VOICE_WEIGHT=0.55
FACE_WEIGHT=0.45
ALLOWED_ORIGINS=https://mindcarex.vercel.app,http://localhost:5173
```

---

## Running Locally

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn main:app --reload --port 8002
```

Verify:
```bash
curl http://localhost:8002/health
# {"status":"ok","service":"svc3_camera_analysis","hf_face_model":true}

curl http://localhost:8002/api/camera/status
# {"hf_face_model":true,"model":"trpakov/vit-face-expression"}
```

Test frame upload:
```bash
# Start a session
curl -X POST http://localhost:8002/api/camera/session/start \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"P001","label":"Test","voice_session_id":null}'

# Upload a face image
curl -X POST http://localhost:8002/api/camera/SESSION_ID/frame \
  -F "file=@face.jpg"
```

---

## Docker

```bash
docker build -t mindcarex-svc3 .
docker run -p 8002:8002 --env-file .env mindcarex-svc3
```

---

## Notes

- **HuggingFace token type**: Must be Fine-grained with "Make calls to serverless Inference Providers" permission. Standard read tokens do not work with `router.huggingface.co`.
- **First call latency**: HF free tier cold-starts the model in ~20s. Subsequent calls take 1–3s. The service retries 3 times on 503.
- **Frame format**: Send JPEG with `Content-Type: image/jpeg`. PNG also works.
- **No face detected**: If no face is visible in the frame, `face_detected: false` is returned and the frame is still recorded (for detection rate tracking).
- **Combined score**: Requires `voice_session_id` to be passed when starting the camera session. The combined endpoint aligns frames to voice chunks by timestamp (±8 second tolerance).
- **Render free tier**: ~50 MB RAM (just HTTP calls, no local model). Fits easily on free tier.
