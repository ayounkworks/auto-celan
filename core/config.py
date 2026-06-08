# ============================================================
# core/config.py
# BASELINE — setelah rollback
#
# Dihapus (tidak dipakai lagi setelah rollback):
#   - DETECT_PAD          (untuk bubble type detection)
#   - BUBBLE_EXPAND_DARK  (untuk _expand_bbox_to_bubble)
#   - DIALOG_BG_MAX_STD   (untuk is_sfx)
#   - DIALOG_BG_LIGHT     (untuk is_sfx)
#   - DIALOG_BG_DARK      (untuk is_sfx)
#   - SFX_MIN_AREA_RATIO  (untuk is_sfx)
#   - SFX_BOX_HEIGHT_MIN  (untuk is_sfx)
#   - SFX_AREA_PER_CHAR   (untuk is_sfx)
#   - SFX_VOTE_THRESHOLD  (untuk is_sfx)
#   - SOLID_FILL_STD_THRESHOLD (untuk solid_fill_inpaint)
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
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_WIDTH        = 1024

# Padding flat untuk expand bbox sebelum masuk lama_mask
# 30px: cukup untuk konteks LaMa tanpa menelan artwork sekitar bubble
PADDING          = 30

# Inpainting crop
INPAINT_CROP_PAD = 128
INPAINT_MIN_RATIO = 0.005

# Batas pixel crop sebelum resize ke RunPod (~1MP aman untuk LaMa)
MAX_RUNPOD_PIXELS = 512 * 2048

# Nilai yang masih dipakai di pipeline lama (kompatibilitas)
MAX_AREA_RATIO     = 0.03
BORDER_SAMPLE      = 18
VARIANCE_THRESHOLD = 300

# Auto-delete output setelah selesai (menit)
OUTPUT_DELETE_DELAY_MINUTES = 15