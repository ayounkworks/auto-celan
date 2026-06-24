# ============================================================
# core/config.py
# YOLO edition — tambah YOLO_MODEL_PATH
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys & Tokens ─────────────────────────────────────
GOOGLE_API_KEY         = os.getenv("GOOGLE_VISION_API_KEY")   # tidak dipakai lagi, keep for compat
DRIVE_OUTPUT_FOLDER_ID = os.getenv("DRIVE_OUTPUT_FOLDER_ID")
RUNPOD_API_KEY         = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID     = os.getenv("RUNPOD_ENDPOINT_ID", "3wjmfk65eoyfd8")

# ── YOLO Detection Model ──────────────────────────────────
# Path ke best.pt
# Di Railway: set env var YOLO_MODEL_PATH ke path di container
# Default: models/best.pt (relatif ke root project)
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best.pt")
)

# ── Image Processing ──────────────────────────────────────
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_WIDTH        = 1024

# Padding flat untuk expand bbox sebelum masuk lama_mask
PADDING          = 20

# Inpainting crop
INPAINT_CROP_PAD  = 128
INPAINT_MIN_RATIO = 0.005

# Batas pixel crop sebelum resize ke RunPod (~1MP aman untuk LaMa)
MAX_RUNPOD_PIXELS = 512 * 2048

# Kompatibilitas pipeline lama
MAX_AREA_RATIO     = 0.03
BORDER_SAMPLE      = 18
VARIANCE_THRESHOLD = 300

# Auto-delete output setelah selesai (menit)
OUTPUT_DELETE_DELAY_MINUTES = 15