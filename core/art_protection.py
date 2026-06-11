# ============================================================
# core/art_protection.py
# FIXED v4:
# - BUG1: _is_dark_bubble_region() lebih longgar (130→150, 0.03→0.01)
# - BUG1: Guard hangul ≥3 karakter = langsung return False (dialog pasti)
# - BUG6: _sample_border() di-cache, tidak dipanggil 2x
# - Threshold tetap 5, semua early exit diperkuat
# ============================================================

import numpy as np
from typing import Tuple

ART_PROTECT_THRESHOLD = 5
BUBBLE_WHITE_MIN    = 210
BUBBLE_BLACK_MAX    = 45
BUBBLE_GRAY_MAX_STD = 30.0
ART_BORDER_PAD      = 20
BUBBLE_BRIGHT_RATIO = 0.55
ART_MIN_AREA_RATIO  = 0.0003
GRADIENT_STD_LOW    = 18.0
GRADIENT_STD_HIGH   = 72.0


def _sample_border(img_np, x1, y1, x2, y2, pad):
    h, w  = img_np.shape[:2]
    bx1, by1 = max(0,x1-pad), max(0,y1-pad)
    bx2, by2 = min(w,x2+pad), min(h,y2+pad)
    strips = []
    if by1 < y1: strips.append(img_np[by1:y1, bx1:bx2])
    if y2 < by2: strips.append(img_np[y2:by2, bx1:bx2])
    if bx1 < x1: strips.append(img_np[y1:y2, bx1:x1])
    if x2 < bx2: strips.append(img_np[y1:y2, x2:bx2])
    if not strips:
        return np.array([])
    return np.concatenate(
        [s.reshape(-1,s.shape[2]) for s in strips if s.size>0 and s.ndim==3], axis=0
    )


def _is_dark_bubble_region(img_np, x1, y1, x2, y2):
    """
    FIXED v4: threshold naik 130→150, bright_ratio turun 0.03→0.01.
    Dark circular bubble (manhwa Korea) → bukan art, jangan protect.
    """
    roi = img_np[y1:y2, x1:x2]
    if roi.size == 0 or roi.mean() > 150:   # FIXED: 130 → 150
        return False
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 20 or roi_w < 20:
        return False
    center = roi[roi_h//4:3*roi_h//4, roi_w//4:3*roi_w//4]
    if center.size == 0:
        return False
    return float((center > 150).mean()) > 0.01  # FIXED: 160→150, 0.03→0.01


def _is_ornate_frame_bubble(img_np, x1, y1, x2, y2):
    """Ornate/decorative frame bubble (pink/gold border) → bukan art."""
    roi = img_np[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 40 or roi_w < 40:
        return False
    my, mx = max(4,roi_h//8), max(4,roi_w//8)
    interior = roi[my:roi_h-my, mx:roi_w-mx]
    if interior.size == 0:
        return False
    return float((interior > 235).all(axis=2).mean()) > 0.65


def _is_gradient_bubble(border_px):
    if border_px.size == 0:
        return False, 128.0
    gray = border_px.mean(axis=1)
    mean, std = float(gray.mean()), float(gray.std())
    if mean > BUBBLE_WHITE_MIN and std < BUBBLE_GRAY_MAX_STD:
        return True, mean
    if mean < BUBBLE_BLACK_MAX:
        return True, mean
    if mean < 160 and GRADIENT_STD_LOW < std < GRADIENT_STD_HIGH:
        return True, mean
    return False, mean


def _has_complex_colored_background(img_np, x1, y1, x2, y2):
    """Dark region butuh threshold jauh lebih tinggi."""
    p = 4
    region = img_np[min(y1+p,y2-1):max(y1+p+1,y2-p),
                    min(x1+p,x2-1):max(x1+p+1,x2-p)]
    if region.size == 0:
        return False
    mean_brightness = float(region.mean())
    avg_std = float(np.stack([
        np.std(region[:,:,0]), np.std(region[:,:,1]), np.std(region[:,:,2])
    ]).mean())
    rg = float(np.mean(np.abs(region[:,:,0].astype(float)-region[:,:,1].astype(float))))
    rb = float(np.mean(np.abs(region[:,:,0].astype(float)-region[:,:,2].astype(float))))
    color_div = (rg + rb) / 2
    if mean_brightness < 80:
        return avg_std > 90.0 and color_div > 45.0
    return avg_std > 70.0 and color_div > 28.0


def _is_decorative_text(text_str):
    t = text_str.strip().replace(" ","").replace("\n","")
    if not t: return False
    if t.isdigit() and len(t) <= 2: return True
    if len(t) == 1 and not t.isalnum(): return True
    return False


def _aspect_ratio_ok(box_w, box_h):
    if box_h == 0: return True
    return 0.15 <= box_w/box_h <= 12.0


def is_art_text(img_np, x1, y1, x2, y2, text_str=""):
    h, w     = img_np.shape[:2]
    total_px = h * w
    box_area = (x2-x1) * (y2-y1)

    if box_area / total_px < ART_MIN_AREA_RATIO:
        return False

    # ── FIXED BUG1: Teks Korea panjang = dialog, bukan art ──
    t_clean = text_str.strip().replace(" ","").replace("\n","")
    hangul_count = sum(1 for c in t_clean if '\uAC00' <= c <= '\uD7A3')
    if hangul_count >= 3:
        return False  # Teks Korea ≥3 char = pasti dialog

    # ── Early exits: jelas bukan art ─────────────────────
    if _is_dark_bubble_region(img_np, x1, y1, x2, y2):
        return False
    if _is_ornate_frame_bubble(img_np, x1, y1, x2, y2):
        return False

    # ── FIXED BUG6: Cache border_px, jangan panggil 2x ──
    border_px = _sample_border(img_np, x1, y1, x2, y2, ART_BORDER_PAD)
    if border_px.size > 0:
        border_mean = float(border_px.mean(axis=1).mean())
        p = 4
        interior = img_np[min(y1+p,y2-1):max(y1+p+1,y2-p),
                          min(x1+p,x2-1):max(x1+p+1,x2-p)]
        int_mean = float(interior.mean()) if interior.size > 0 else 128.0
        if int_mean < 80 or border_mean < 50:
            return False

    # ── Scoring ──────────────────────────────────────────
    x_center = (x1+x2)/2
    is_side  = (x_center < w*0.25) or (x_center > w*0.75)
    score    = 0

    if _has_complex_colored_background(img_np, x1, y1, x2, y2):
        if not is_side:
            score += 1

    # FIXED BUG6: gunakan border_px yang sudah di-cache
    if border_px.size > 0:
        is_bubble, _ = _is_gradient_bubble(border_px)
        if not is_bubble:
            score += 1

    if _is_decorative_text(text_str):
        score += 1
    if not _aspect_ratio_ok(x2-x1, y2-y1):
        score += 1
    if box_area/total_px < 0.002 and score >= 1:
        score += 1
    if box_area/total_px > 0.06:
        score += 2

    protected = score >= ART_PROTECT_THRESHOLD
    if protected:
        print(f"  [art_protect] Skip '{text_str[:20]}' score={score} box=({x1},{y1},{x2},{y2})")
    return protected