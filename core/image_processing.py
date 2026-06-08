# ============================================================
# core/image_processing.py
# BASELINE — rollback dari versi overengineered
#
# Pipeline:
#   Vision annotations
#   → _merge_text_blocks()   gabung box berdekatan
#   → smart_clean()          build lama_mask flat (PADDING per box)
#   → get_inpaint_crop()     crop bounding box seluruh mask
#
# Semua filter dan guard DIHAPUS:
#   - dark/white bubble detection
#   - large bubble validation
#   - SFX filter
#   - art protection
#   - bubble expansion (dark/white)
#   - dark_bubble_gradient_fill
#   - solid_fill_inpaint
#   - mask morphology (MinFilter/MaxFilter)
# ============================================================

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Optional, Tuple
from dataclasses import dataclass

from core.config import (
    PADDING,
    MAX_AREA_RATIO,
    INPAINT_CROP_PAD,
)


# ── Utilities ─────────────────────────────────────────────

def to_bytes(img, fmt="JPEG", quality=90) -> io.BytesIO:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt, **({"quality": quality} if fmt == "JPEG" else {}))
    buffer.seek(0)
    return buffer


def progress_bar(completed, total, width=20) -> str:
    if total == 0:
        return "`[--------------------] 0% (0/0)`"
    percent = completed / total
    filled  = int(width * percent)
    if filled == 0:       bar = "-" * width
    elif filled == width: bar = "=" * width
    else:                 bar = "=" * (filled - 1) + ">" + "-" * (width - filled)
    return f"`[{bar}] {int(percent*100)}% ({completed}/{total})`"


def format_eta(seconds) -> str:
    if seconds <= 0: return "sebentar lagi"
    if seconds < 60: return f"~{seconds}s"
    return f"~{seconds//60}m {seconds%60}s"


def get_dynamic_batch_size(total_files: int) -> int:
    if total_files <= 5:    return total_files
    elif total_files <= 20: return 12
    elif total_files <= 50: return 20
    else:                   return 25


# ── Text block merging ────────────────────────────────────

@dataclass
class MergedText:
    description: str
    x1: int; y1: int
    x2: int; y2: int


def _merge_text_blocks(texts, width, height, threshold=40) -> list:
    """
    Gabungkan bounding box Vision API yang berdekatan.
    threshold=40: box dengan jarak < 40px dianggap satu bubble.
    Nilai lebih kecil dari 70 sebelumnya — mencegah bubble berbeda
    di panel berbeda ikut di-merge jadi satu mask besar.
    """
    if not texts or len(texts) <= 1:
        return []

    items = []
    for text in texts[1:]:
        v = text.bounding_poly.vertices
        items.append(MergedText(
            description=text.description,
            x1=min(p.x for p in v), y1=min(p.y for p in v),
            x2=max(p.x for p in v), y2=max(p.y for p in v),
        ))

    items.sort(key=lambda i: (i.y1, i.x1))

    merged = True
    while merged:
        merged    = False
        new_items = []
        while items:
            curr           = items.pop(0)
            found_neighbor = False
            for i, other in enumerate(new_items):
                v_threshold = int(threshold * 1.5)
                h_threshold = threshold
                overlap = not (
                    curr.x1 > other.x2 + h_threshold or
                    curr.x2 < other.x1 - h_threshold or
                    curr.y1 > other.y2 + v_threshold or
                    curr.y2 < other.y1 - v_threshold
                )
                if overlap:
                    other.x1 = min(other.x1, curr.x1)
                    other.y1 = min(other.y1, curr.y1)
                    other.x2 = max(other.x2, curr.x2)
                    other.y2 = max(other.y2, curr.y2)
                    other.description += " " + curr.description
                    found_neighbor     = True
                    merged             = True
                    break
            if not found_neighbor:
                new_items.append(curr)
        items = new_items

    return items


# ── Inpaint crop ──────────────────────────────────────────

def get_inpaint_crop(
    img:  Image.Image,
    mask: Image.Image,
    pad:  int = INPAINT_CROP_PAD,
) -> Optional[Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]]:
    """
    Crop area inpainting berdasarkan bounding box seluruh mask.
    Menambahkan pad di semua sisi untuk konteks LaMa.
    Returns (img_crop, mask_crop, (l, t, r, b)) atau None.
    """
    bbox = mask.getbbox()
    if not bbox:
        return None

    w, h       = img.size
    l, t, r, b = bbox

    l = max(0, l - pad)
    t = max(0, t - pad)
    r = min(w, r + pad)
    b = min(h, b + pad)

    return img.crop((l, t, r, b)), mask.crop((l, t, r, b)), (l, t, r, b)


# ── Validate inpaint result ───────────────────────────────

def validate_inpaint(inpaint, img_crop) -> Optional[Image.Image]:
    """
    Guard terhadap output RunPod yang corrupt.
    Cek: tidak None, ukuran sesuai, tidak hitam total, tidak flat.
    """
    if inpaint is None:
        return None
    if inpaint.size != img_crop.size:
        inpaint = inpaint.resize(img_crop.size, Image.LANCZOS)
    inpaint = inpaint.convert("RGB")
    arr = np.array(inpaint)
    if float(np.mean(arr)) < 5.0:
        print("[validate_inpaint] Hasil hitam, skip")
        return None
    if float(np.std(arr)) < 2.0:
        print("[validate_inpaint] Hasil uniform/corrupt, skip")
        return None
    return inpaint


# ── Smart clean — BASELINE ────────────────────────────────

def smart_clean(original, texts, img_np):
    """
    BASELINE pipeline:
      1. Merge text blocks yang berdekatan
      2. Expand setiap box dengan PADDING flat
      3. Tulis ke lama_mask

    Tidak ada filter, tidak ada guard, tidak ada special case.
    Semua text yang terdeteksi Vision masuk ke mask.
    LaMa yang memutuskan cara fill terbaik.

    Returns:
        result        — image (tidak dimodifikasi, identik dengan original)
        lama_mask     — PIL Image mode "L", putih di area teks
        sfx_count     — selalu 0 (kompatibilitas dengan pipeline.py)
        dialog_count  — jumlah boxes yang masuk mask
        inpaint_boxes — list koordinat boxes (untuk logging)
    """
    width, height = original.size
    result        = original.copy().convert("RGB")
    lama_mask     = Image.new("L", (width, height), 0)
    lama_draw     = ImageDraw.Draw(lama_mask)

    merged_texts  = _merge_text_blocks(texts, width, height)
    inpaint_boxes = []
    dialog_count  = 0

    for text in merged_texts:
        # Expand dengan PADDING flat di semua sisi
        x1 = max(0,      text.x1 - PADDING)
        y1 = max(0,      text.y1 - PADDING)
        x2 = min(width,  text.x2 + PADDING)
        y2 = min(height, text.y2 + PADDING)

        if x1 >= x2 or y1 >= y2:
            continue

        lama_draw.rectangle([x1, y1, x2, y2], fill=255)
        inpaint_boxes.append((x1, y1, x2, y2))
        dialog_count += 1

    print(f"  [smart_clean] merged={len(merged_texts)} "
          f"masked={dialog_count} "
          f"image={width}x{height}")

    # sfx_count=0 untuk kompatibilitas return signature
    return result, lama_mask, 0, dialog_count, inpaint_boxes