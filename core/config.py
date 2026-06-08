# ============================================================
# core/config.py
# FIXED v3:
# - PADDING naik 25 → 80 (cover dark circular bubble)
# - BUBBLE_EXPAND_DARK: padding ekstra untuk dark bubble
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys & Tokens ─────────────────────────────────────
GOOGLE_API_KEY         = os.getenv("GOOGLE_VISION_API_KEY")
DRIVE_OUTPUT_FOLDER_ID = os.getenv("DRIVE_OUTPUT_FOLDER_ID")
RUNPOD_API_KEY         = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID     = os.getenv("RUNPOD_ENDPOINT_ID", "3wjmfk65eoyfd8")

# ── Image Processing ──────────────────────────────────────
VALID_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp"}
MAX_WIDTH          = 1024

# FIXED v3: naik dari 25 → 80
# Dark circular bubble: diameter 150-400px, teks hanya di tengah ~100px
# Gap dari teks ke tepi bubble = 50-125px per sisi
# Padding 80px cukup untuk most cases
PADDING            = 80

# Padding ekstra khusus untuk dark circular bubble
# Dipakai di smart_clean() saat bubble terdeteksi dark
BUBBLE_EXPAND_DARK = 120

MAX_AREA_RATIO     = 0.03
BORDER_SAMPLE      = 18
VARIANCE_THRESHOLD = 300

# SFX filter thresholds
DIALOG_BG_MAX_STD   = 55.0
DIALOG_BG_LIGHT     = 165
DIALOG_BG_DARK      = 35

SFX_MIN_AREA_RATIO  = 0.0003
SFX_BOX_HEIGHT_MIN  = 40
SFX_AREA_PER_CHAR   = 4000
SFX_VOTE_THRESHOLD  = 2

# Inpainting crop
INPAINT_CROP_PAD    = 150
INPAINT_MIN_RATIO   = 0.005

# Solid fill detection
SOLID_FILL_STD_THRESHOLD = 28.0

# Auto-delete output setelah selesai (menit)
OUTPUT_DELETE_DELAY_MINUTES = 15