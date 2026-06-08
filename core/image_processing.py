# ============================================================
# core/image_processing.py
# v6 — Integrasikan improvement dari patched version (zip):
#
# [IP-1] solid_fill_inpaint(): skip RunPod untuk region uniform
#        Bubble hitam/putih solid → fill avg_color langsung (~10x lebih cepat)
#        Diambil dari zip [PROD-5] + ditingkatkan
#
# [IP-2] get_inpaint_crop(): adaptive pad berdasarkan ukuran crop
#        Crop besar tidak perlu banyak context → kurangi VRAM
#        Diambil dari zip [FIX-3/FIX-10]
#
# [IP-3] is_sfx() hangul repetition: deteksi SFX Korea yang repeating
#        Misal "드드드드" atau "쾅쾅" = SFX, bukan dialog
#        Diambil dari zip is_sfx() + enhancement untuk manhwa
#
# [IP-4] smart_clean() [FIX-BOX]: large bubble validation via bg complexity
#        Box > MAX_AREA_RATIO boleh lolos kalau background solid
#        Sebelumnya: langsung skip → banyak bubble besar terlewat
#
# [IP-5] lama_mask cleanup: MinFilter + MaxFilter + GaussianBlur
#        Sudah ada di v5, dipertahankan + diperkuat
#
# [IP-6] _merge_text_blocks(): threshold dinaikkan 55 → 70 untuk
#        manhwa yang teks-nya lebih tersebar
#
# Workflow TETAP sama dengan v5:
#   RAW bbox → deteksi → expand → fill (gradient/LaMa)
# ============================================================

import io
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageStat
from typing import Optional, Tuple

from core.config import (
    PADDING, DETECT_PAD, MAX_AREA_RATIO, BORDER_SAMPLE, VARIANCE_THRESHOLD,
    DIALOG_BG_MAX_STD, DIALOG_BG_LIGHT, DIALOG_BG_DARK,
    SFX_MIN_AREA_RATIO, SFX_BOX_HEIGHT_MIN, SFX_AREA_PER_CHAR, SFX_VOTE_THRESHOLD,
    INPAINT_CROP_PAD, INPAINT_MIN_RATIO, BUBBLE_EXPAND_DARK,
    SOLID_FILL_STD_THRESHOLD, MAX_RUNPOD_PIXELS,
)
from core.art_protection import is_art_text
from dataclasses import dataclass


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


# ── [IP-1] Solid Fill Inpaint ─────────────────────────────
# Deteksi region uniform/solid sebelum kirim ke RunPod.
# Bubble putih/hitam solid → langsung fill warna rata → ~10x lebih cepat,
# dan hasilnya lebih bersih dari LaMa untuk area yang memang uniform.

def solid_fill_inpaint(
    prefilled:  Image.Image,
    mask_crop:  Image.Image,
    img_crop:   Image.Image,
    cl: int,    ct: int,
) -> Optional[Image.Image]:
    """
    Cek apakah area di bawah mask seragam/solid.
    Jika iya: fill dengan warna rata-rata background → return Image.
    Jika tidak: return None → caller harus pakai RunPod.

    Cara kerja:
    1. Ambil pixel DI LUAR mask pada crop (= area background yang tidak tertutup teks)
    2. Hitung std warna area tersebut
    3. Jika std rendah → background seragam → fill warna rata
    """
    arr      = np.array(img_crop.convert("RGB"))
    mask_arr = np.array(mask_crop.convert("L"))
    h, w     = mask_arr.shape

    # Dilasi mask untuk mencari area background "di sekitar" teks
    pad = min(20, h // 4, w // 4)
    if pad > 0:
        mask_pil     = Image.fromarray(mask_arr)
        mask_dilated = mask_pil.filter(ImageFilter.MaxFilter(size=pad * 2 + 1))
        border_area  = np.array(mask_dilated) == 0  # area luar mask setelah dilasi
    else:
        border_area = mask_arr == 0

    if border_area.sum() < 100:
        return None  # tidak cukup sample background

    bg_pixels = arr[border_area]
    bg_mean   = bg_pixels.mean(axis=0)
    bg_std    = float(bg_pixels.std())

    # Background terlalu kompleks → biarkan RunPod handle
    if bg_std > SOLID_FILL_STD_THRESHOLD:
        return None

    avg_color = tuple(int(c) for c in bg_mean[:3])

    # Fill area mask pada full image dengan avg_color
    result     = np.array(prefilled.convert("RGB"))
    h_full, w_full = result.shape[:2]

    # Project mask_crop ke koordinat penuh (cl, ct)
    my1 = ct; my2 = min(h_full, ct + h)
    mx1 = cl; mx2 = min(w_full, cl + w)
    ch  = my2 - my1; cw = mx2 - mx1

    if ch <= 0 or cw <= 0:
        return None

    mask_slice = mask_arr[:ch, :cw]
    mask_bool  = mask_slice > 128
    result[my1:my2, mx1:mx2][mask_bool] = avg_color

    # Soft blend edge agar tidak terlalu sharp
    result_img  = Image.fromarray(result)
    soft_mask   = Image.fromarray(mask_arr).filter(ImageFilter.GaussianBlur(3))
    sm_arr      = np.array(soft_mask) / 255.0

    orig_arr    = np.array(prefilled.convert("RGB"))
    blended     = orig_arr.copy().astype(float)
    blended[my1:my2, mx1:mx2] = (
        sm_arr[:ch, :cw, np.newaxis] * np.array(avg_color, dtype=float)
        + (1 - sm_arr[:ch, :cw, np.newaxis]) * orig_arr[my1:my2, mx1:mx2].astype(float)
    )
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


# ── [IP-2] Adaptive Inpaint Crop ─────────────────────────

def get_inpaint_crop(
    img:  Image.Image,
    mask: Image.Image,
    pad:  int = INPAINT_CROP_PAD,
) -> Optional[Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]]:
    """
    Crop area inpainting dengan adaptive pad berdasarkan ukuran crop.
    Crop besar tidak butuh banyak konteks tambahan → kurangi pad → hemat VRAM.
    """
    bbox = mask.getbbox()
    if not bbox:
        return None

    w, h       = img.size
    l, t, r, b = bbox

    # Adaptive pad: makin besar crop → makin kecil pad
    bbox_area  = (r - l) * (b - t)
    page_area  = w * h
    crop_ratio = bbox_area / page_area

    if crop_ratio > 0.4:
        pad = min(pad, 32)
    elif crop_ratio > 0.2:
        pad = min(pad, 64)

    l = max(0, l - pad)
    t = max(0, t - pad)
    r = min(w, r + pad)
    b = min(h, b + pad)

    return img.crop((l, t, r, b)), mask.crop((l, t, r, b)), (l, t, r, b)


# ── Dark Bubble Gradient Fill ─────────────────────────────

def dark_bubble_gradient_fill(
    img:    Image.Image,
    img_np: np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
) -> Optional[Image.Image]:
    """
    Fill dark circular bubble dengan gradient warna dari luar bubble.
    Return None kalau tidak bisa → fallback ke RunPod.
    """
    h_img, w_img = img_np.shape[:2]
    sample_h     = min(15, max(5, (y2 - y1) // 10))

    top_region = img_np[max(0, y1 - sample_h):y1, max(0, x1):min(w_img, x2)]
    bot_region = img_np[y2:min(h_img, y2 + sample_h), max(0, x1):min(w_img, x2)]

    if top_region.size == 0 and bot_region.size == 0:
        return None

    top_color = top_region.reshape(-1, 3).mean(axis=0) if top_region.size > 0 else None
    bot_color = bot_region.reshape(-1, 3).mean(axis=0) if bot_region.size > 0 else None

    top_bright = float(top_color.mean()) if top_color is not None else 0
    bot_bright = float(bot_color.mean()) if bot_color is not None else 0

    if top_bright < 60 and bot_bright < 60:
        return None  # luar juga gelap → biarkan RunPod

    if top_color is None: top_color = bot_color
    if bot_color is None: bot_color = top_color

    result  = img.copy()
    res_arr = np.array(result)

    ay1 = max(0, y1); ay2 = min(h_img, y2)
    ax1 = max(0, x1); ax2 = min(w_img, x2)
    gh = ay2 - ay1; gw = ax2 - ax1
    if gh <= 0 or gw <= 0:
        return None

    t_values   = np.linspace(0, 1, gh).reshape(-1, 1)
    gradient   = (1 - t_values) * top_color + t_values * bot_color
    gradient   = np.clip(gradient, 0, 255).astype(np.uint8)
    gradient2d = np.tile(gradient, (1, gw, 1))
    res_arr[ay1:ay2, ax1:ax2] = gradient2d[:gh, :gw]
    result = Image.fromarray(res_arr)

    # Soft blend di edge
    pad  = min(15, max(5, (y2 - y1) // 8))
    bx1  = max(0, x1 - pad); by1 = max(0, y1 - pad)
    bx2  = min(img.width, x2 + pad); by2 = min(img.height, y2 + pad)
    patch = result.crop((bx1, by1, bx2, by2))
    orig  = img.crop((bx1, by1, bx2, by2))
    mask  = Image.new("L", patch.size, 0)
    md    = ImageDraw.Draw(mask)
    md.rectangle([x1 - bx1, y1 - by1, x2 - bx1, y2 - by1], fill=255)
    mask  = mask.filter(ImageFilter.GaussianBlur(pad * 0.6))
    result.paste(Image.composite(patch, orig, mask), (bx1, by1))
    return result


# ── Bubble Expansion ──────────────────────────────────────

def _expand_bbox_to_bubble(img_np, x1, y1, x2, y2, max_expand=200):
    h, w      = img_np.shape[:2]
    gray      = img_np.mean(axis=2)
    threshold = 80

    new_y1 = y1
    for dy in range(1, max_expand):
        row_y = max(0, y1 - dy)
        row   = gray[row_y, max(0, x1):min(w, x2)]
        if row.size > 0 and row.mean() < threshold: new_y1 = row_y
        else: break

    new_y2 = y2
    for dy in range(1, max_expand):
        row_y = min(h - 1, y2 + dy)
        row   = gray[row_y, max(0, x1):min(w, x2)]
        if row.size > 0 and row.mean() < threshold: new_y2 = row_y
        else: break

    new_x1 = x1
    for dx in range(1, max_expand):
        col_x = max(0, x1 - dx)
        col   = gray[max(0, new_y1):min(h, new_y2), col_x]
        if col.size > 0 and col.mean() < threshold: new_x1 = col_x
        else: break

    new_x2 = x2
    for dx in range(1, max_expand):
        col_x = min(w - 1, x2 + dx)
        col   = gray[max(0, new_y1):min(h, new_y2), col_x]
        if col.size > 0 and col.mean() < threshold: new_x2 = col_x
        else: break

    return max(0, new_x1 - 5), max(0, new_y1 - 5), min(w - 1, new_x2 + 5), min(h - 1, new_y2 + 5)


def _expand_bbox_to_white_bubble(img_np, x1, y1, x2, y2, max_expand=60):
    h, w      = img_np.shape[:2]
    gray      = img_np.mean(axis=2)
    threshold = 180

    ny1 = y1
    for dy in range(1, max_expand):
        row = gray[max(0, y1 - dy), max(0, x1):min(w, x2)]
        if row.size > 0 and row.mean() > threshold: ny1 = max(0, y1 - dy)
        else: break
    ny2 = y2
    for dy in range(1, max_expand):
        row = gray[min(h - 1, y2 + dy), max(0, x1):min(w, x2)]
        if row.size > 0 and row.mean() > threshold: ny2 = min(h - 1, y2 + dy)
        else: break
    nx1 = x1
    for dx in range(1, max_expand):
        col = gray[ny1:ny2, max(0, x1 - dx)]
        if col.size > 0 and col.mean() > threshold: nx1 = max(0, x1 - dx)
        else: break
    nx2 = x2
    for dx in range(1, max_expand):
        col = gray[ny1:ny2, min(w - 1, x2 + dx)]
        if col.size > 0 and col.mean() > threshold: nx2 = min(w - 1, x2 + dx)
        else: break

    return max(0, nx1 - 5), max(0, ny1 - 5), min(w - 1, nx2 + 5), min(h - 1, ny2 + 5)


# ── [IP-6] Text Merging ───────────────────────────────────

@dataclass
class MergedText:
    description: str
    x1: int; y1: int
    x2: int; y2: int


def _merge_text_blocks(texts, width, height, threshold=70) -> list:
    """
    [IP-6] threshold naik 55 → 70: manhwa teks lebih tersebar per bubble.
    Gabungkan bounding box yang berdekatan agar satu bubble = satu inpaint job.
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
            curr             = items.pop(0)
            found_neighbor   = False
            for i, other in enumerate(new_items):
                v_threshold = int(threshold * 1.5)
                h_threshold = threshold
                if not (curr.x1 > other.x2 + h_threshold or curr.x2 < other.x1 - h_threshold
                        or curr.y1 > other.y2 + v_threshold or curr.y2 < other.y1 - v_threshold):
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


# ── Bubble Detection ──────────────────────────────────────

def _is_dark_bubble(img_np, x1, y1, x2, y2):
    roi = img_np[y1:y2, x1:x2]
    if roi.size == 0: return False
    if float(roi.mean()) > 140: return False
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 10 or roi_w < 10: return False
    return float((roi > 150).mean()) > 0.01


def _is_white_bubble(img_np, x1, y1, x2, y2):
    roi = img_np[y1:y2, x1:x2]
    if roi.size == 0: return False
    return float(roi.mean()) > 180


# ── [IP-3] SFX Filter ────────────────────────────────────

def is_sfx(img_np, rx1, ry1, rx2, ry2, text_str="") -> bool:
    """
    [IP-3] Voting system + hangul repetition detection.
    Tambahan vs v5:
    - Repetisi hangul yang lebih akurat (sliding window)
    - Threshold DIALOG_BG_DARK naik ke 70 (dari config v4)
    - SFX_MIN_AREA_RATIO naik ke 0.001 (dari config v4)
    """
    box_w    = rx2 - rx1
    box_h    = ry2 - ry1
    total_px = img_np.shape[0] * img_np.shape[1]

    if (box_w * box_h) / total_px < SFX_MIN_AREA_RATIO:
        return False

    region = img_np[max(0, ry1):min(img_np.shape[0], ry2),
                    max(0, rx1):min(img_np.shape[1], rx2)]

    if region.size > 0:
        gray   = np.mean(region, axis=2) if region.ndim == 3 else region.astype(float)
        mean_b = float(np.mean(gray))
        std_b  = float(np.std(gray))

        if mean_b >= DIALOG_BG_LIGHT:
            return False
        if mean_b <= DIALOG_BG_DARK:
            return False
        if std_b <= DIALOG_BG_MAX_STD and mean_b > DIALOG_BG_DARK:
            return False

    score = 0

    if (box_w * box_h) / total_px > SFX_MIN_AREA_RATIO * 3:
        score += 1
    if box_h >= SFX_BOX_HEIGHT_MIN:
        score += 1

    t = text_str.strip().replace(" ", "").replace("\n", "")
    if len(t) > 0:
        area_per_char = (box_w * box_h) / len(t)
        if area_per_char > SFX_AREA_PER_CHAR:
            score += 1
        if t.isupper() and t.isalpha() and len(t) <= 8:
            score += 1

        # Katakana SFX
        katakana = sum(1 for c in t if '\u30A0' <= c <= '\u30FF')
        if len(t) > 0 and katakana / len(t) > 0.7:
            score += 1

        # [IP-3] Hangul SFX detection — lebih akurat
        hangul = [c for c in t if '\uAC00' <= c <= '\uD7A3']
        if hangul:
            hangul_ratio = len(hangul) / len(t)
            if hangul_ratio > 0.5:
                # Repetisi karakter (드드드, 쾅쾅쾅)
                half = len(t) // 2
                if half >= 1 and len(t) >= 2 and t[:half] == t[half:half * 2]:
                    score += 2  # bobot tinggi — jelas SFX berulang
                elif len(t) <= 4 and (box_w * box_h) / total_px < SFX_MIN_AREA_RATIO * 5:
                    score += 1  # hangul pendek di area kecil

        # Ellipsis / titik = SFX (jeda narasi)
        if re.fullmatch(r'[.…·・]+', t):
            score += 2

    return score >= SFX_VOTE_THRESHOLD


# ── Validate Inpaint ──────────────────────────────────────

def validate_inpaint(inpaint, img_crop) -> Optional[Image.Image]:
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


# ── [IP-4] Smart Clean ────────────────────────────────────

def smart_clean(original, texts, img_np):
    """
    [IP-4] Tambah large-bubble validation via background complexity:
    Box > MAX_AREA_RATIO → cek background solid/kompleks.
    - Solid bg (std<45 + mean ekstrem) = bubble besar valid → proses
    - Background kompleks (artwork) → skip

    Workflow:
    1. RAW bbox (+ DETECT_PAD kecil) untuk DETECTION
    2. Expand ke full bubble bounds untuk FILL
    3. dark bubble → gradient fill; white bubble → LaMa mask
    """
    width, height = original.size
    total_area    = width * height
    result        = original.copy().convert("RGB")
    draw          = ImageDraw.Draw(result)
    lama_mask     = Image.new("L", (width, height), 0)
    lama_draw     = ImageDraw.Draw(lama_mask)
    current_np    = img_np.copy()

    merged_texts  = _merge_text_blocks(texts, width, height)
    inpaint_boxes = []
    sfx_count     = 0
    dialog_count  = 0

    # ── INSTRUMENTATION counters ──────────────────────────
    _elim = {
        "invalid_bbox":    0,
        "large_bubble":    0,
        "sfx":             0,
        "art":             0,
        "invalid_fill":    0,
        "bad_ratio":       0,
        "dark_gradient":   0,
        "dark_lama":       0,
        "dialog":          0,
    }
    # ─────────────────────────────────────────────────────

    for text in merged_texts:
        # RAW bbox + DETECT_PAD kecil untuk deteksi
        rx1 = max(0, text.x1 - DETECT_PAD)
        ry1 = max(0, text.y1 - DETECT_PAD)
        rx2 = min(width,  text.x2 + DETECT_PAD)
        ry2 = min(height, text.y2 + DETECT_PAD)

        if rx1 >= rx2 or ry1 >= ry2:
            # [1] bbox invalid
            _elim["invalid_bbox"] += 1
            print(f"  [ELIM-1:invalid_bbox] '{text.description[:20]}' "
                  f"box=({rx1},{ry1},{rx2},{ry2})")
            continue

        box_w = rx2 - rx1
        box_h = ry2 - ry1

        is_dark_bub  = _is_dark_bubble(current_np, rx1, ry1, rx2, ry2)
        is_white_bub = _is_white_bubble(current_np, rx1, ry1, rx2, ry2)

        # [IP-4] Large bubble validation
        if (box_w * box_h) > total_area * MAX_AREA_RATIO:
            if not is_dark_bub and not is_white_bub:
                # Bukan bubble jelas → cek bg complexity
                pad_s  = max(4, min(box_w, box_h) // 8)
                ix1    = min(rx1 + pad_s, rx2 - 1)
                iy1    = min(ry1 + pad_s, ry2 - 1)
                ix2    = max(ix1 + 1, rx2 - pad_s)
                iy2    = max(iy1 + 1, ry2 - pad_s)
                region = current_np[iy1:iy2, ix1:ix2]
                if region.size > 0:
                    gray        = np.mean(region, axis=2) if region.ndim == 3 else region.astype(float)
                    bg_std      = float(np.std(gray))
                    bg_mean     = float(np.mean(gray))
                    is_solid_bg = bg_std < 45 and (bg_mean > 160 or bg_mean < 60)
                    if not is_solid_bg:
                        # [2] large bubble validation gagal
                        _elim["large_bubble"] += 1
                        print(f"  [ELIM-2:large_bubble] '{text.description[:20]}' "
                              f"bg_std={bg_std:.1f} bg_mean={bg_mean:.1f} "
                              f"box_area_ratio={(box_w*box_h)/total_area:.4f}")
                        continue  # background kompleks (artwork) -> skip
                else:
                    _elim["large_bubble"] += 1
                    print(f"  [ELIM-2:large_bubble] '{text.description[:20]}' "
                          f"region kosong")
                    continue

        # SFX check (hanya kalau bukan bubble jelas)
        if not is_dark_bub and not is_white_bub:
            if is_sfx(current_np, rx1, ry1, rx2, ry2, text.description):
                sfx_count += 1
                _elim["sfx"] += 1
                print(f"  [ELIM-3:sfx] '{text.description[:20]}' "
                      f"box=({rx1},{ry1},{rx2},{ry2}) "
                      f"size={box_w}x{box_h}")
                continue

        # Art protection
        if not is_dark_bub and not is_white_bub:
            if is_art_text(current_np, rx1, ry1, rx2, ry2, text.description):
                _elim["art"] += 1
                print(f"  [ELIM-4:art] '{text.description[:20]}' "
                      f"box=({rx1},{ry1},{rx2},{ry2}) "
                      f"size={box_w}x{box_h}")
                continue

        dialog_count += 1

        # Tentukan fill bbox
        if is_dark_bub:
            fx1, fy1, fx2, fy2 = _expand_bbox_to_bubble(
                current_np, rx1, ry1, rx2, ry2, max_expand=BUBBLE_EXPAND_DARK
            )
        elif is_white_bub:
            fx1, fy1, fx2, fy2 = _expand_bbox_to_white_bubble(
                current_np, rx1, ry1, rx2, ry2
            )
        else:
            SMALL_PAD = 30
            fx1 = max(0, text.x1 - SMALL_PAD)
            fy1 = max(0, text.y1 - SMALL_PAD)
            fx2 = min(width,  text.x2 + SMALL_PAD)
            fy2 = min(height, text.y2 + SMALL_PAD)

        fx1 = max(0, fx1); fy1 = max(0, fy1)
        fx2 = min(width, fx2); fy2 = min(height, fy2)

        if fx1 >= fx2 or fy1 >= fy2:
            # [5] fill bbox invalid setelah expand
            _elim["invalid_fill"] += 1
            print(f"  [ELIM-5:invalid_fill] '{text.description[:20]}' "
                  f"fill_box=({fx1},{fy1},{fx2},{fy2})")
            continue

        fw = fx2 - fx1; fh = fy2 - fy1
        if fw > 0 and fh > 0:
            ratio = fw / fh
            if ratio < 0.05 or ratio > 20.0:
                # [6] aspect ratio invalid
                _elim["bad_ratio"] += 1
                print(f"  [ELIM-6:bad_ratio] '{text.description[:20]}' "
                      f"ratio={ratio:.2f} fill_box=({fx1},{fy1},{fx2},{fy2})")
                continue

        # Fill
        if is_dark_bub:
            filled = dark_bubble_gradient_fill(result, current_np, fx1, fy1, fx2, fy2)
            if filled is not None:
                result     = filled
                draw       = ImageDraw.Draw(result)
                current_np = np.array(result)
                _elim["dark_gradient"] += 1
                print(f"  [ELIM-7a:dark_gradient] '{text.description[:20]}' "
                      f"fill_box=({fx1},{fy1},{fx2},{fy2})")
                continue
            # Fallback: LaMa
            _elim["dark_lama"] += 1
            print(f"  [ELIM-7b:dark_lama] '{text.description[:20]}' "
                  f"fill_box=({fx1},{fy1},{fx2},{fy2})")
            lama_draw.rectangle([fx1, fy1, fx2, fy2], fill=255)
            continue

        lama_draw.rectangle([fx1, fy1, fx2, fy2], fill=255)
        inpaint_boxes.append((fx1, fy1, fx2, fy2))
        _elim["dialog"] += 1

    # ── INSTRUMENTATION summary ───────────────────────────
    total_merged = len(merged_texts)
    total_elim   = sum(v for k, v in _elim.items() if k != "dialog")
    print(f"\n  [ELIM-SUMMARY] merged={total_merged} | "
          f"dialog={_elim['dialog']} | "
          f"dark_gradient={_elim['dark_gradient']} | "
          f"dark_lama={_elim['dark_lama']} | "
          f"sfx={_elim['sfx']} | "
          f"art={_elim['art']} | "
          f"large_bubble={_elim['large_bubble']} | "
          f"invalid_bbox={_elim['invalid_bbox']} | "
          f"invalid_fill={_elim['invalid_fill']} | "
          f"bad_ratio={_elim['bad_ratio']} | "
          f"total_eliminated={total_elim}\n")
    # ─────────────────────────────────────────────────────

    # [IP-5] Cleanup lama_mask: hapus noise kecil, tutup celah, perhalus tepi
    if lama_mask.getbbox():
        lama_mask = lama_mask.filter(ImageFilter.MinFilter(3))   # hapus noise kecil
        lama_mask = lama_mask.filter(ImageFilter.MaxFilter(7))   # tutup celah teks
        lama_mask = lama_mask.filter(ImageFilter.GaussianBlur(radius=3))  # perhalus tepi

    return result, lama_mask, sfx_count, dialog_count, inpaint_boxes