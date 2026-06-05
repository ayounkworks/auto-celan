# ============================================================
# core/image_processing.py
# smart_clean, SFX filter, solid fill, validate_inpaint, crop
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
    INPAINT_CROP_PAD, SOLID_FILL_STD_THRESHOLD,
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
    if filled == 0:
        bar = "-" * width
    elif filled == width:
        bar = "=" * width
    else:
        bar = "=" * (filled - 1) + ">" + "-" * (width - filled)
    return f"`[{bar}] {int(percent * 100)}% ({completed}/{total})`"


def format_eta(seconds) -> str:
    if seconds <= 0:
        return "sebentar lagi"
    if seconds < 60:
        return f"~{seconds}s"
    return f"~{seconds // 60}m {seconds % 60}s"


def get_dynamic_batch_size(total_files: int) -> int:
    if total_files <= 5:
        return total_files
    elif total_files <= 20:
        return 12
    elif total_files <= 50:
        return 20
    else:
        return 25


# ── SFX Filter ────────────────────────────────────────────

def is_sfx(img_np: np.ndarray, x1: int, y1: int, x2: int, y2: int,
           text_str: str = "") -> bool:
    """Voting system: 4 sinyal ringan, >= SFX_VOTE_THRESHOLD → SFX → skip."""
    box_w    = x2 - x1
    box_h    = y2 - y1
    total_px = img_np.shape[0] * img_np.shape[1]

    if (box_w * box_h) / total_px < SFX_MIN_AREA_RATIO:
        return False

    margin_x = max(2, box_w // 5)
    margin_y = max(2, box_h // 5)
    ix1 = min(x1 + margin_x, x2 - 1)
    iy1 = min(y1 + margin_y, y2 - 1)
    ix2 = max(ix1 + 1, x2 - margin_x)
    iy2 = max(iy1 + 1, y2 - margin_y)

    region = img_np[iy1:iy2, ix1:ix2]
    if region.size > 0:
        gray            = np.mean(region, axis=2) if region.ndim == 3 else region.astype(float)
        mean_brightness = float(np.mean(gray))
        std_brightness  = float(np.std(gray))

        if mean_brightness >= DIALOG_BG_LIGHT:
            return False
        # FIX Bug1: DIALOG_BG_DARK sekarang 35 (dari 70).
        # Dark bubble solid (mean < 35) bukan SFX — return False langsung.
        if mean_brightness <= DIALOG_BG_DARK:
            return False
        if std_brightness <= DIALOG_BG_MAX_STD:
            return False

    score = 0
    if (box_w * box_h) / total_px > SFX_MIN_AREA_RATIO * 3:
        score += 1
    if box_h >= SFX_BOX_HEIGHT_MIN:
        score += 1

    t          = text_str.strip().replace(" ", "").replace("\n", "")
    char_count = len(t)
    if char_count > 0:
        area_per_char = (box_w * box_h) / char_count
        if area_per_char > SFX_AREA_PER_CHAR:
            score += 1
        if t.isupper() and t.isalpha() and len(t) <= 8:
            score += 1
        katakana = sum(1 for c in t if '\u30A0' <= c <= '\u30FF')
        if katakana / len(t) > 0.7:
            score += 1

        # FIX: Korean onomatopoeia / SFX detection
        # Korean SFX tidak punya uppercase atau katakana — deteksi via pola:
        # 1. Repeating syllable: 다르다르, 넝실넝실, 뿔뿔 → SFX
        # 2. Short Korean text (≤4 syllable) di atas art background
        # 3. Ellipsis/dots only: .... → SFX
        hangul_chars = [c for c in t if '\uAC00' <= c <= '\uD7A3']
        hangul_ratio = len(hangul_chars) / len(t) if len(t) > 0 else 0

        if hangul_ratio > 0.5:
            # Cek repeating syllable pattern (SFX khas Korean)
            # Pattern: AB repeated (다르다르, 넝실넝실)
            half = len(t) // 2
            if half >= 1 and len(t) >= 2 and t[:half] == t[half:half*2]:
                score += 2   # repeating = almost certainly SFX
            # Short Korean (≤4 chars) dengan box kecil = SFX di atas art
            elif len(t) <= 4 and (box_w * box_h) / total_px < SFX_MIN_AREA_RATIO * 5:
                score += 1
            # Dots/ellipsis only
        if re.fullmatch(r'[.…·・]+', t):
            score += 2

    return score >= SFX_VOTE_THRESHOLD


# ── Crop for Inpaint ──────────────────────────────────────



# ── Validate Inpaint Result ───────────────────────────────

def validate_inpaint(inpaint: Image.Image, img_crop: Image.Image) -> Optional[Image.Image]:
    """Return None kalau hasil inpaint corrupt, hitam, atau uniform."""
    if inpaint is None:
        return None

    if inpaint.size != img_crop.size:
        inpaint = inpaint.resize(img_crop.size, Image.LANCZOS)

    inpaint = inpaint.convert("RGB")
    arr     = np.array(inpaint)

    if float(np.mean(arr)) < 5.0:
        print("[validate_inpaint] Hasil hitam terdeteksi, skip inpaint")
        return None

    if float(np.std(arr)) < 2.0:
        print("[validate_inpaint] Hasil uniform/corrupt terdeteksi, skip inpaint")
        return None

    return inpaint


# ── Smart Clean ───────────────────────────────────────────

def smart_clean(
    original: Image.Image,
    texts,
    img_np:   np.ndarray
) -> Tuple[Image.Image, Image.Image, int, int]:
    """
    Proses text annotations Vision API.
    Returns: (result_img, lama_mask, sfx_count, dialog_count)
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
            pad_s  = max(4, min(box_w, box_h) // 8)
            ix1    = min(x1 + pad_s, x2 - 1)
            iy1    = min(y1 + pad_s, y2 - 1)
            ix2    = max(ix1 + 1, x2 - pad_s)
            iy2    = max(iy1 + 1, y2 - pad_s)
            region = img_np[iy1:iy2, ix1:ix2]
            if region.size > 0:
                gray       = np.mean(region, axis=2) if region.ndim == 3 else region.astype(float)
                bg_std     = float(np.std(gray))
                bg_mean    = float(np.mean(gray))
                is_solid_bg = bg_std < 45 and (bg_mean > 160 or bg_mean < 60)
                if not is_solid_bg:
                    continue
            else:
                continue

        if is_sfx(img_np, x1, y1, x2, y2, text_str):
            sfx_count += 1
            continue

        # ── Art Protection ────────────────────────────────
        # Cek apakah teks ini bagian dari art (kaos, sign, background)
        # yang tidak boleh dihapus. Dijalankan SETELAH filter SFX
        # agar SFX sudah ter-skip duluan.
        if is_art_text(img_np, x1, y1, x2, y2, text_str):
            continue  # lindungi — jangan masukkan ke mask

        dialog_count += 1

        ratio = box_w / box_h if box_h > 0 else 1.0
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
        strips = [s for s in crops if s.size[0] > 0 and s.size[1] > 0]

        if not strips:
            continue

        total_w  = sum(s.size[0] for s in strips)
        combined = Image.new("RGB", (total_w, max(s.size[1] for s in strips)), 0)
        offset   = 0
        for s in strips:
            combined.paste(s, (offset, 0))
            offset += s.size[0]

        stat     = ImageStat.Stat(combined)
        variance = sum(stat.var[:3]) / 3

        if variance < VARIANCE_THRESHOLD:
            blur_pad   = max(BORDER_SAMPLE, 20)
            blur_x1    = max(0, x1 - blur_pad)
            blur_y1    = max(0, y1 - blur_pad)
            blur_x2    = min(width,  x2 + blur_pad)
            blur_y2    = min(height, y2 + blur_pad)
            blur_patch = original.crop((blur_x1, blur_y1, blur_x2, blur_y2))
            blurred    = blur_patch.filter(ImageFilter.GaussianBlur(12))
            rel_x      = x1 - blur_x1
            rel_y      = y1 - blur_y1
            fill_patch = blurred.crop((rel_x, rel_y, rel_x + box_w, rel_y + box_h))
            if fill_patch.size == (box_w, box_h):
                result.paste(fill_patch, (x1, y1))
            else:
                avg_color = tuple(int(c) for c in stat.mean[:3])
                draw.rectangle([x1, y1, x2, y2], fill=avg_color)
        else:
            lama_draw.rectangle([x1, y1, x2, y2], fill=255)

    return result, lama_mask, sfx_count, dialog_count


# ── Solid Fill (skip RunPod untuk area polos) ─────────────

def solid_fill_inpaint(
    prefilled: Image.Image,
    mask:      Image.Image,
    img_crop:  Image.Image,
    cl: int, ct: int
) -> Optional[Image.Image]:
    """
    Kalau area yang di-mask terdeteksi solid/uniform → fill langsung.
    Return None kalau background kompleks (harus pakai RunPod/LaMa).
    """
    arr      = np.array(img_crop)
    mask_arr = np.array(mask.convert("L"))

    h, w  = mask_arr.shape
    pad   = min(20, h // 4, w // 4)

    if pad > 0:
        mask_pil     = Image.fromarray(mask_arr)
        mask_dilated = mask_pil.filter(ImageFilter.MaxFilter(size=pad * 2 + 1))
        border_area  = np.array(mask_dilated) == 0
    else:
        border_area = mask_arr == 0

    if border_area.sum() < 100:
        return None

    bg_pixels = arr[border_area]
    bg_mean   = bg_pixels.mean(axis=0)
    bg_std    = bg_pixels.std()

    if bg_std > SOLID_FILL_STD_THRESHOLD:
        return None

    # FIX: Jangan solid-fill kalau area mask sendiri punya texture/gradient
    # Bubble putih webtoon sering punya shadow/gradient di dalam
    mask_interior = arr[mask_arr > 128] if mask_arr.sum() > 0 else np.array([])
    if mask_interior.size > 0:
        interior_std = float(mask_interior.std())
        # Kalau interior punya variasi tinggi → ada konten (background art, gradient)
        # → jangan solid fill, biarkan RunPod yang handle
        if interior_std > 40:
            return None

    avg_color = tuple(int(c) for c in bg_mean[:3])
    result    = prefilled.copy()

    mask_pil_full     = Image.new("L", prefilled.size, 0)
    mask_pil_full.paste(mask.convert("L"), (cl, ct))

    result_arr    = np.array(result)
    mask_full_arr = np.array(mask_pil_full)
    mask_bool     = mask_full_arr > 128

    result_arr[mask_bool] = avg_color
    return Image.fromarray(result_arr)