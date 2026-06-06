# ============================================================
# core/image_processing.py
# FIXED v3:
# - import re top-level (fix UnboundLocalError)
# - PADDING naik 25→80 (dari config)
# - _expand_bbox_to_bubble(): auto-expand untuk dark/circular bubble
# - is_sfx(): hapus dark bg penalty
# - _is_circular_or_spiky_bubble(): helper baru
# - smart_clean(): force-dialog untuk circular bubble + expand bbox
# ============================================================

import io
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageStat
from typing import Optional, Tuple

from core.config import (
    PADDING, MAX_AREA_RATIO, BORDER_SAMPLE, VARIANCE_THRESHOLD,
    DIALOG_BG_MAX_STD, DIALOG_BG_LIGHT, DIALOG_BG_DARK,
    SFX_MIN_AREA_RATIO, SFX_BOX_HEIGHT_MIN, SFX_AREA_PER_CHAR, SFX_VOTE_THRESHOLD,
    INPAINT_CROP_PAD, SOLID_FILL_STD_THRESHOLD, BUBBLE_EXPAND_DARK,
)
from core.art_protection import is_art_text


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
    if filled == 0:    bar = "-" * width
    elif filled == width: bar = "=" * width
    else: bar = "=" * (filled-1) + ">" + "-" * (width-filled)
    return f"`[{bar}] {int(percent*100)}% ({completed}/{total})`"


def format_eta(seconds) -> str:
    if seconds <= 0: return "sebentar lagi"
    if seconds < 60: return f"~{seconds}s"
    return f"~{seconds//60}m {seconds%60}s"


def get_dynamic_batch_size(total_files: int) -> int:
    if total_files <= 5:   return total_files
    elif total_files <= 20: return 12
    elif total_files <= 50: return 20
    else: return 25


# ── Bubble Expansion ──────────────────────────────────────

def _expand_bbox_to_bubble(img_np: np.ndarray,
                            x1: int, y1: int, x2: int, y2: int,
                            max_expand: int = 150) -> Tuple[int,int,int,int]:
    """
    Expand bounding box untuk cover seluruh dark circular bubble.
    Cara kerja: flood-fill ke luar dari bbox selama pixel gelap.
    Dipakai untuk dark bubble agar seluruh area ter-mask, bukan hanya teks.
    """
    h, w = img_np.shape[:2]
    gray = img_np.mean(axis=2)

    # Threshold: pixel dianggap bagian dari dark bubble
    threshold = 80

    # Expand ke atas
    new_y1 = y1
    for dy in range(1, max_expand):
        row_y = max(0, y1 - dy)
        row   = gray[row_y, max(0,x1):min(w,x2)]
        if row.size > 0 and row.mean() < threshold:
            new_y1 = row_y
        else:
            break

    # Expand ke bawah
    new_y2 = y2
    for dy in range(1, max_expand):
        row_y = min(h-1, y2 + dy)
        row   = gray[row_y, max(0,x1):min(w,x2)]
        if row.size > 0 and row.mean() < threshold:
            new_y2 = row_y
        else:
            break

    # Expand ke kiri
    new_x1 = x1
    for dx in range(1, max_expand):
        col_x = max(0, x1 - dx)
        col   = gray[max(0,new_y1):min(h,new_y2), col_x]
        if col.size > 0 and col.mean() < threshold:
            new_x1 = col_x
        else:
            break

    # Expand ke kanan
    new_x2 = x2
    for dx in range(1, max_expand):
        col_x = min(w-1, x2 + dx)
        col   = gray[max(0,new_y1):min(h,new_y2), col_x]
        if col.size > 0 and col.mean() < threshold:
            new_x2 = col_x
        else:
            break

    return new_x1, new_y1, new_x2, new_y2


# ── Circular/Spiky Bubble Detection ──────────────────────

def _is_circular_or_spiky_bubble(img_np: np.ndarray,
                                   x1: int, y1: int,
                                   x2: int, y2: int) -> bool:
    """
    Deteksi speech bubble berbentuk circular atau spiky/star.
    - Dark circular bubble: bg gelap, teks terang
    - Spiky/star bubble: interior putih, outline tidak beraturan
    Return True = ini bubble, bukan art.
    """
    roi = img_np[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    roi_h, roi_w = roi.shape[:2]
    if roi_h < 30 or roi_w < 30:
        return False

    aspect = roi_w / max(roi_h, 1)
    if not (0.4 < aspect < 2.5):
        return False

    margin   = max(5, min(roi_h,roi_w)//8)
    interior = roi[margin:roi_h-margin, margin:roi_w-margin]
    if interior.size == 0:
        return False

    interior_mean = float(interior.mean())
    full_mean     = float(roi.mean())
    contrast      = abs(interior_mean - full_mean)

    # Spiky white bubble
    interior_white = float((interior > 220).all(axis=2).mean())
    if interior_white > 0.55 and contrast > 10.0:
        return True

    # Dark circular bubble
    if interior_mean < 100:
        bright_ratio = float((interior > 160).mean())
        if bright_ratio > 0.02 and contrast > 8.0:
            return True

    return False


# ── SFX Filter ────────────────────────────────────────────

def is_sfx(img_np: np.ndarray, x1: int, y1: int, x2: int, y2: int,
           text_str: str = "") -> bool:
    """
    FIXED v3: Hapus dark bg penalty sepenuhnya.
    Dark background bukan alasan skip SFX detection.
    """
    box_w    = x2 - x1
    box_h    = y2 - y1
    total_px = img_np.shape[0] * img_np.shape[1]

    if (box_w * box_h) / total_px < SFX_MIN_AREA_RATIO:
        return False

    margin_x = max(2, box_w//5)
    margin_y = max(2, box_h//5)
    ix1 = min(x1+margin_x, x2-1)
    iy1 = min(y1+margin_y, y2-1)
    ix2 = max(ix1+1, x2-margin_x)
    iy2 = max(iy1+1, y2-margin_y)

    region = img_np[iy1:iy2, ix1:ix2]
    if region.size > 0:
        gray = np.mean(region, axis=2) if region.ndim==3 else region.astype(float)
        mean_b = float(np.mean(gray))
        std_b  = float(np.std(gray))

        if mean_b >= DIALOG_BG_LIGHT:
            return False
        # FIXED: hapus early return untuk dark bg
        # Dark background BUKAN alasan skip
        if std_b <= DIALOG_BG_MAX_STD and mean_b > DIALOG_BG_DARK:
            return False

    score = 0
    if (box_w*box_h)/total_px > SFX_MIN_AREA_RATIO*3: score += 1
    if box_h >= SFX_BOX_HEIGHT_MIN: score += 1

    t = text_str.strip().replace(" ","").replace("\n","")
    char_count = len(t)
    if char_count > 0:
        area_per_char = (box_w*box_h)/char_count
        if area_per_char > SFX_AREA_PER_CHAR: score += 1
        if t.isupper() and t.isalpha() and len(t) <= 8: score += 1
        katakana = sum(1 for c in t if '\u30A0' <= c <= '\u30FF')
        if katakana/len(t) > 0.7: score += 1

        hangul = [c for c in t if '\uAC00' <= c <= '\uD7A3']
        hangul_ratio = len(hangul)/len(t)

        if hangul_ratio > 0.5:
            half = len(t)//2
            if half >= 1 and len(t) >= 2 and t[:half] == t[half:half*2]:
                score += 2
            elif len(t) <= 4 and (box_w*box_h)/total_px < SFX_MIN_AREA_RATIO*5:
                score += 1

        # FIXED: re sudah di-import di top level
        if re.fullmatch(r'[.…·・]+', t):
            score += 2

    return score >= SFX_VOTE_THRESHOLD


# ── Validate Inpaint ──────────────────────────────────────

def validate_inpaint(inpaint: Image.Image,
                     img_crop: Image.Image) -> Optional[Image.Image]:
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


# ── Smart Clean ───────────────────────────────────────────

def smart_clean(
    original: Image.Image,
    texts,
    img_np:   np.ndarray
) -> Tuple[Image.Image, Image.Image, int, int]:
    """
    FIXED v3:
    - _is_circular_or_spiky_bubble() force-dialog
    - _expand_bbox_to_bubble() untuk dark bubble agar full bubble ter-mask
    - PADDING sudah naik via config (25→80)
    """
    width, height = original.size
    total_area    = width * height
    result        = original.copy().convert("RGB")
    draw          = ImageDraw.Draw(result)
    lama_mask     = Image.new("L", (width, height), 0)
    lama_draw     = ImageDraw.Draw(lama_mask)

    sfx_count    = 0
    dialog_count = 0

    for text in texts[1:]:
        vertices = text.bounding_poly.vertices
        xs = [v.x for v in vertices]
        ys = [v.y for v in vertices]

        # Base bbox dengan PADDING (sudah naik ke 80 via config)
        x1 = max(0, min(xs) - PADDING)
        y1 = max(0, min(ys) - PADDING)
        x2 = min(width,  max(xs) + PADDING)
        y2 = min(height, max(ys) + PADDING)

        if x1 >= x2 or y1 >= y2:
            continue

        box_w    = x2 - x1
        box_h    = y2 - y1
        text_str = text.description if hasattr(text, 'description') else ""

        if box_w * box_h > total_area * MAX_AREA_RATIO:
            pad_s  = max(4, min(box_w,box_h)//8)
            ix1 = min(x1+pad_s, x2-1)
            iy1 = min(y1+pad_s, y2-1)
            ix2 = max(ix1+1, x2-pad_s)
            iy2 = max(iy1+1, y2-pad_s)
            region = img_np[iy1:iy2, ix1:ix2]
            if region.size > 0:
                gray   = np.mean(region,axis=2) if region.ndim==3 else region.astype(float)
                bg_std  = float(np.std(gray))
                bg_mean = float(np.mean(gray))
                if not (bg_std < 45 and (bg_mean > 160 or bg_mean < 60)):
                    continue
            else:
                continue

        # ── Cek circular/spiky bubble DULU ───────────────
        force_dialog = _is_circular_or_spiky_bubble(img_np, x1, y1, x2, y2)

        # Kalau dark bubble, expand bbox ke seluruh bubble area
        is_dark_bubble = False
        if force_dialog:
            roi = img_np[y1:y2, x1:x2]
            if roi.size > 0 and roi.mean() < 100:
                is_dark_bubble = True
                x1, y1, x2, y2 = _expand_bbox_to_bubble(
                    img_np, x1, y1, x2, y2,
                    max_expand=BUBBLE_EXPAND_DARK
                )
                # Clamp
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

        if not force_dialog:
            if is_sfx(img_np, x1, y1, x2, y2, text_str):
                sfx_count += 1
                continue
            if is_art_text(img_np, x1, y1, x2, y2, text_str):
                continue

        dialog_count += 1

        ratio = (x2-x1) / max(y2-y1, 1)
        if ratio < 0.1 or ratio > 10.0:
            continue

        bx1 = max(0, x1 - BORDER_SAMPLE)
        by1 = max(0, y1 - BORDER_SAMPLE)
        bx2 = min(width,  x2 + BORDER_SAMPLE)
        by2 = min(height, y2 + BORDER_SAMPLE)

        crops = [
            original.crop((bx1, by1, bx2, y1)),
            original.crop((bx1, y2,  bx2, by2)),
            original.crop((bx1, y1,  x1,  y2)),
            original.crop((x2,  y1,  bx2, y2)),
        ]
        strips = [s for s in crops if s.size[0]>0 and s.size[1]>0]

        if not strips:
            continue

        total_w  = sum(s.size[0] for s in strips)
        combined = Image.new("RGB", (total_w, max(s.size[1] for s in strips)), 0)
        offset   = 0
        for s in strips:
            combined.paste(s, (offset,0))
            offset += s.size[0]

        stat     = ImageStat.Stat(combined)
        variance = sum(stat.var[:3]) / 3

        if variance < VARIANCE_THRESHOLD:
            blur_pad = max(BORDER_SAMPLE, 20)
            bpx1 = max(0, x1-blur_pad)
            bpy1 = max(0, y1-blur_pad)
            bpx2 = min(width,  x2+blur_pad)
            bpy2 = min(height, y2+blur_pad)
            patch   = original.crop((bpx1,bpy1,bpx2,bpy2))
            blurred = patch.filter(ImageFilter.GaussianBlur(12))
            rel_x   = x1 - bpx1
            rel_y   = y1 - bpy1
            bw, bh  = x2-x1, y2-y1
            fill    = blurred.crop((rel_x,rel_y,rel_x+bw,rel_y+bh))
            if fill.size == (bw, bh):
                result.paste(fill, (x1, y1))
            else:
                avg_color = tuple(int(c) for c in stat.mean[:3])
                draw.rectangle([x1,y1,x2,y2], fill=avg_color)
        else:
            lama_draw.rectangle([x1,y1,x2,y2], fill=255)

    return result, lama_mask, sfx_count, dialog_count


# ── Solid Fill ────────────────────────────────────────────

def solid_fill_inpaint(
    prefilled: Image.Image,
    mask:      Image.Image,
    img_crop:  Image.Image,
    cl: int, ct: int
) -> Optional[Image.Image]:
    arr      = np.array(img_crop)
    mask_arr = np.array(mask.convert("L"))
    h, w     = mask_arr.shape
    pad      = min(20, h//4, w//4)

    if pad > 0:
        mask_pil     = Image.fromarray(mask_arr)
        mask_dilated = mask_pil.filter(ImageFilter.MaxFilter(size=pad*2+1))
        border_area  = np.array(mask_dilated) == 0
    else:
        border_area  = mask_arr == 0

    if border_area.sum() < 100:
        return None

    bg_pixels = arr[border_area]
    bg_mean   = bg_pixels.mean(axis=0)
    bg_std    = bg_pixels.std()

    if bg_std > SOLID_FILL_STD_THRESHOLD:
        return None

    mask_interior = arr[mask_arr > 128] if mask_arr.sum() > 0 else np.array([])
    if mask_interior.size > 0 and float(mask_interior.std()) > 40:
        return None

    avg_color = tuple(int(c) for c in bg_mean[:3])
    result    = prefilled.copy()

    mask_full = Image.new("L", prefilled.size, 0)
    mask_full.paste(mask.convert("L"), (cl, ct))

    res_arr  = np.array(result)
    mf_arr   = np.array(mask_full)
    res_arr[mf_arr > 128] = avg_color
    return Image.fromarray(res_arr)