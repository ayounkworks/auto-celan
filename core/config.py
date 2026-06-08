# ============================================================
# core/config.py
# v4 — Integrasikan improvement dari patched version (zip):
#   [C-1] DIALOG_BG_DARK naik 35 → 70 (sama dengan zip P-4)
#         Sebelumnya terlalu rendah, banyak sfx di bg gelap lolos
#   [C-2] SFX_MIN_AREA_RATIO naik 0.0003 → 0.001
#         0.0003 terlalu kecil — noise kecil dianggap SFX
#   [C-3] SFX_BOX_HEIGHT_MIN naik 40 → 70 (zip P-6)
#   [C-4] SOLID_FILL_STD_THRESHOLD naik 28 → 48 (zip OPT-7)
#         Manhwa bg putih std ~25-40, threshold 28 miss banyak kasus
#   [C-5] INPAINT_CROP_PAD turun 150 → 128 (zip FIX-3)
#         Kurangi VRAM ~25% tanpa kehilangan kualitas
#   [C-6] MAX_REPLICATE_PIXELS: batas pixel crop sebelum resize ke RunPod
#   [C-7] DETECT_PAD: pisahkan padding untuk deteksi vs fill
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

# PADDING untuk fill bubble (bukan untuk deteksi)
# DETECT_PAD lebih kecil, dipakai untuk deteksi tipe bubble
PADDING            = 80
DETECT_PAD         = 15   # [C-7] padding kecil khusus untuk deteksi bubble/sfx/art

# Padding ekstra khusus untuk dark circular bubble
BUBBLE_EXPAND_DARK = 120

MAX_AREA_RATIO     = 0.03
BORDER_SAMPLE      = 18
VARIANCE_THRESHOLD = 300

# SFX filter thresholds — v4 tuning
DIALOG_BG_MAX_STD   = 55.0
DIALOG_BG_LIGHT     = 165
DIALOG_BG_DARK      = 70     # [C-1] naik dari 35 → 70

SFX_MIN_AREA_RATIO  = 0.001  # [C-2] naik dari 0.0003 → 0.001
SFX_BOX_HEIGHT_MIN  = 70     # [C-3] naik dari 40 → 70
SFX_AREA_PER_CHAR   = 4000
SFX_VOTE_THRESHOLD  = 2

# Inpainting crop
INPAINT_CROP_PAD    = 128    # [C-5] turun dari 150 → 128
INPAINT_MIN_RATIO   = 0.005

# Solid fill detection — deteksi area uniform sebelum kirim ke RunPod
# [C-4] naik dari 28 → 48: manhwa bg putih std ~25-40, harus catch semua
SOLID_FILL_STD_THRESHOLD = 48.0

# Batas pixel crop yang dikirim ke RunPod sebelum di-resize
# Mencegah OOM untuk crop sangat besar. ~1MP aman untuk LaMa
MAX_RUNPOD_PIXELS = 512 * 2048  # [C-6] ~1MP

# Auto-delete output setelah selesai (menit)
OUTPUT_DELETE_DELAY_MINUTES = 15