import os
from dotenv import load_dotenv

load_dotenv()

# ── Required ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", "")

# ── Optional — falls back to expression-only mode if not set ─────────────────
# Same token used in svc1 — no extra cost
HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")

# Combined score weights (voice + face)
VOICE_WEIGHT: float = float(os.getenv("VOICE_WEIGHT", "0.55"))
FACE_WEIGHT:  float = float(os.getenv("FACE_WEIGHT",  "0.45"))

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL not set.\n"
        "Add your Neon connection string:\n"
        "  postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require"
    )
