# ============================================================
# core/config.py
# Semua konstanta, env vars, dan konfigurasi global
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys & Tokens ─────────────────────────────────────
GOOGLE_API_KEY         = os.getenv("GOOGLE_VISION_API_KEY")
DRIVE_OUTPUT_FOLDER_ID = os.getenv("DRIVE_OUTPUT_FOLDER_ID")

# FIX: RUNPOD_API_KEY sekarang dibaca dari .env (sebelumnya tidak ada os.getenv)
RUNPOD_API_KEY         = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID     = os.getenv("RUNPOD_ENDPOINT_ID", "3wjmfk65eoyfd8")

# ── Image Processing ──────────────────────────────────────
VALID_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp"}
MAX_WIDTH          = 1024

PADDING            = 25
MAX_AREA_RATIO     = 0.03
BORDER_SAMPLE      = 18
VARIANCE_THRESHOLD = 300

# SFX filter thresholds
DIALOG_BG_MAX_STD   = 55.0
DIALOG_BG_LIGHT     = 165
DIALOG_BG_DARK      = 70
SFX_MIN_AREA_RATIO  = 0.001

# SFX voting
SFX_BOX_HEIGHT_MIN  = 70
SFX_AREA_PER_CHAR   = 4000
SFX_VOTE_THRESHOLD  = 2

# Inpainting crop
INPAINT_CROP_PAD    = 180
INPAINT_MIN_RATIO   = 0.005

# Solid fill detection
# Lebih toleran untuk bubble transparan/gradient
SOLID_FILL_STD_THRESHOLD = 45.0

# ── Credit & Job Rules ────────────────────────────────────
