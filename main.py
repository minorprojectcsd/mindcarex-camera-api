"""
SVC 3 — Camera / Face Expression Analysis
Run:    uvicorn main:app --host 0.0.0.0 --port 8002 --reload
Docker: see Dockerfile
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db
from app.router   import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()   # creates camera_sessions + camera_frames tables in Neon
    yield


app = FastAPI(
    title="Camera Face Analysis — SVC 3",
    version="1.0.0",
    description=(
        "Real-time facial expression and stress analysis from video frames. "
        "Uses HuggingFace trpakov/vit-face-expression model via Inference API."
    ),
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # must be False when using list with Flask
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/camera")


@app.get("/health")
def health():
    from app.frame_analyzer import HF_API_TOKEN
    return {
        "status":        "ok",
        "service":       "svc3_camera_analysis",
        "hf_face_model": bool(HF_API_TOKEN),
    }
