# ============================================================
# core/image_processing.py
# FIXED v5 - WORKFLOW REWRITE:
#
# Bug logika yang diperbaiki:
# L1: Detection (bubble/SFX/art) sekarang pakai RAW bbox (Vision bbox)
#     bukan padded bbox. Padding hanya untuk fill area.
# L2: is_sfx + is_art_text pakai raw bbox → lebih akurat
# L3: _expand_bbox_to_bubble mulai dari raw bbox (teks), bukan padded
# L4: current_np di-update setelah setiap fill
# L5: dark bubble → gradient fill (bukan LaMa) → tidak lagi blob hitam
#
# Semua filter TETAP ADA tapi dipanggil dengan koordinat yang benar.
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
    INPAINT_CROP_PAD, BUBBLE_EXPAND_DARK,
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


# ── Dark Bubble Gradient Fill ─────────────────────────────

def dark_bubble_gradient_fill(
    img:    Image.Image,
    img_np: np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
) -> Optional[Image.Image]:
    """
    Fill dark circular bubble dengan gradient warna dari luar bubble.
    Hasilnya natural, tidak ada blob hitam seperti LaMa.
    Return None kalau tidak bisa (luar juga gelap) → fallback ke LaMa.
    """
    h_img, w_img = img_np.shape[:2]
    sample_h = min(15, max(5, (y2 - y1) // 10))

    top_region = img_np[max(0, y1-sample_h):y1, max(0,x1):min(w_img,x2)]
    bot_region = img_np[y2:min(h_img, y2+sample_h), max(0,x1):min(w_img,x2)]

    if top_region.size == 0 and bot_region.size == 0:
        return None

    top_color = top_region.reshape(-1, 3).mean(axis=0) if top_region.size > 0 else None
    bot_color = bot_region.reshape(-1, 3).mean(axis=0) if bot_region.size > 0 else None

    top_bright = float(top_color.mean()) if top_color is not None else 0
    bot_bright = float(bot_color.mean()) if bot_color is not None else 0

    # Kalau atas dan bawah keduanya gelap → tidak bisa gradient fill
    if top_bright < 60 and bot_bright < 60:
        return None

    if top_color is None: top_color = bot_color
    if bot_color is None: bot_color = top_color

    result  = img.copy()
    res_arr = np.array(result)

    fill_h = max(1, y2 - y1)
    fill_w = max(1, x2 - x1)
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
    pad  = min(15, max(5, (y2-y1)//8))
    bx1  = max(0, x1-pad); by1 = max(0, y1-pad)
    bx2  = min(img.width, x2+pad); by2 = min(img.height, y2+pad)
    patch = result.crop((bx1, by1, bx2, by2))
    orig  = img.crop((bx1, by1, bx2, by2))
    mask  = Image.new("L", patch.size, 0)
    md    = ImageDraw.Draw(mask)
    md.rectangle([x1-bx1, y1-by1, x2-bx1, y2-by1], fill=255)
    mask  = mask.filter(ImageFilter.GaussianBlur(pad * 0.6))
    result.paste(Image.composite(patch, orig, mask), (bx1, by1))
    return result


# ── Bubble Expansion ──────────────────────────────────────

def _expand_bbox_to_bubble(img_np: np.ndarray,
                            x1: int, y1: int, x2: int, y2: int,
                            max_expand: int = 200) -> Tuple[int,int,int,int]:
    """
    FIXED L3: Mulai dari raw bbox (teks), expand ke luar sampai
    pixel tidak gelap lagi. Threshold 80 = batas dark bubble.
    """
    h, w = img_np.shape[:2]
    gray = img_np.mean(axis=2)
    threshold = 80

    new_y1 = y1
    for dy in range(1, max_expand):
        row_y = max(0, y1 - dy)
        row   = gray[row_y, max(0,x1):min(w,x2)]
        if row.size > 0 and row.mean() < threshold:
            new_y1 = row_y
        else:
            break

    new_y2 = y2
    for dy in range(1, max_expand):
        row_y = min(h-1, y2 + dy)
        row   = gray[row_y, max(0,x1):min(w,x2)]
        if row.size > 0 and row.mean() < threshold:
            new_y2 = row_y
        else:
            break

    new_x1 = x1
    for dx in range(1, max_expand):
        col_x = max(0, x1 - dx)
        col   = gray[max(0,new_y1):min(h,new_y2), col_x]
        if col.size > 0 and col.mean() < threshold:
            new_x1 = col_x
        else:
            break

    new_x2 = x2
    for dx in range(1, max_expand):
        col_x = min(w-1, x2 + dx)
        col   = gray[max(0,new_y1):min(h,new_y2), col_x]
        if col.size > 0 and col.mean() < threshold:
            new_x2 = col_x
        else:
            break

    # Tambahkan safety margin agar mencakup sisa piksel teks di pinggiran
    return max(0, new_x1-5), max(0, new_y1-5), min(w-1, new_x2+5), min(h-1, new_y2+5)


def _expand_bbox_to_white_bubble(img_np: np.ndarray,
                                x1: int, y1: int, x2: int, y2: int,
                                max_expand: int = 60) -> Tuple[int,int,int,int]:
    """
    Ekspansi untuk bubble putih. Berhenti saat menemukan pixel yang 
    cukup gelap (garis tepi bubble).
    """
    h, w = img_np.shape[:2]
    gray = img_np.mean(axis=2)
    threshold = 180 # Cari area yang mulai gelap (tepi bubble)

    ny1 = y1
    for dy in range(1, max_expand):
        row = gray[max(0, y1-dy), max(0,x1):min(w,x2)]
        if row.size > 0 and row.mean() > threshold: ny1 = max(0, y1-dy)
        else: break
    ny2 = y2
    for dy in range(1, max_expand):
        row = gray[min(h-1, y2+dy), max(0,x1):min(w,x2)]
        if row.size > 0 and row.mean() > threshold: ny2 = min(h-1, y2+dy)
        else: break
    nx1 = x1
    for dx in range(1, max_expand):
        col = gray[ny1:ny2, max(0, x1-dx)]
        if col.size > 0 and col.mean() > threshold: nx1 = max(0, x1-dx)
        else: break
    nx2 = x2
    for dx in range(1, max_expand):
        col = gray[ny1:ny2, min(w-1, x2+dx)]
        if col.size > 0 and col.mean() > threshold: nx2 = min(w-1, x2+dx)
        else: break

    # Tambahkan safety margin 5px agar mencakup area anti-aliasing teks
    return max(0, nx1-5), max(0, ny1-5), min(w-1, nx2+5), min(h-1, ny2+5)

# ── Text Merging Logic ────────────────────────────────────

@dataclass
class MergedText:
    description: str
    x1: int
    y1: int
    x2: int
    y2: int

def _merge_text_blocks(texts, width, height, threshold=55) -> list[MergedText]:
    """
    Menggabungkan bounding box yang berdekatan (dalam jarak threshold pixel).
    Sangat penting agar satu bubble tidak terpecah jadi banyak inpaint kecil.
    """
    if not texts or len(texts) <= 1:
        return []

    items = []
    for text in texts[1:]: # Skip index 0 (full text)
        v = text.bounding_poly.vertices
        items.append(MergedText(
            description=text.description,
            x1=min(p.x for p in v), y1=min(p.y for p in v),
            x2=max(p.x for p in v), y2=max(p.y for p in v)
        ))

    # Sort berdasarkan posisi Y lalu X agar penggabungan lebih teratur
    items.sort(key=lambda i: (i.y1, i.x1))

    merged = True
    while merged:
        merged = False
        new_items = []
        while items:
            curr = items.pop(0)
            found_neighbor = False
            for i, other in enumerate(new_items):
                # Cek jika kotak berdekatan (dengan toleransi threshold)
                # Gunakan threshold yang lebih longgar untuk sumbu Y (vertikal) pada manga
                v_threshold = int(threshold * 1.5)
                h_threshold = threshold

                if not (curr.x1 > other.x2 + h_threshold or curr.x2 < other.x1 - h_threshold or
                        curr.y1 > other.y2 + v_threshold or curr.y2 < other.y1 - v_threshold):
                    # Gabungkan koordinat
                    other.x1 = min(other.x1, curr.x1)
                    other.y1 = min(other.y1, curr.y1)
                    other.x2 = max(other.x2, curr.x2)
                    other.y2 = max(other.y2, curr.y2)
                    other.description += " " + curr.description
                    found_neighbor = True
                    merged = True
                    break
            if not found_neighbor:
                new_items.append(curr)
        items = new_items
    return items

# ── Bubble Detection ──────────────────────────────────────

def _is_dark_bubble(img_np: np.ndarray,
                    x1: int, y1: int,
                    x2: int, y2: int) -> bool:
    """
    FIXED L1: Pakai raw bbox (tight sekitar teks) untuk deteksi.
    Dark bubble = bg gelap, ada teks terang di dalam.
    """
    roi = img_np[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    mean_val = float(roi.mean())
    # Background harus gelap
    if mean_val > 140:
        return False
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 10 or roi_w < 10:
        return False
    # Cek ada teks terang (pixel > 150) — artinya teks putih di bg hitam
    bright = float((roi > 150).mean())
    return bright > 0.01


def _is_white_bubble(img_np: np.ndarray,
                     x1: int, y1: int,
                     x2: int, y2: int) -> bool:
    """
    White/light bubble — bg cerah. Selalu hapus (dialog biasa).
    """
    roi = img_np[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    return float(roi.mean()) > 180


# ── SFX Filter ────────────────────────────────────────────

def is_sfx(img_np: np.ndarray,
           rx1: int, ry1: int, rx2: int, ry2: int,
           text_str: str = "") -> bool:
    """
    FIXED L2: Terima raw bbox dari Vision API (bukan padded).
    SFX = sound effect yang seharusnya skip (tidak dihapus).
    """
    box_w    = rx2 - rx1
    box_h    = ry2 - ry1
    total_px = img_np.shape[0] * img_np.shape[1]

    if (box_w * box_h) / total_px < SFX_MIN_AREA_RATIO:
        return False

    region = img_np[max(0,ry1):min(img_np.shape[0],ry2),
                    max(0,rx1):min(img_np.shape[1],rx2)]
    if region.size > 0:
        gray   = np.mean(region, axis=2) if region.ndim==3 else region.astype(float)
        mean_b = float(np.mean(gray))
        std_b  = float(np.std(gray))

        if mean_b >= DIALOG_BG_LIGHT:
            return False
        if std_b <= DIALOG_BG_MAX_STD and mean_b > DIALOG_BG_DARK:
            return False

    score = 0
    if (box_w*box_h)/total_px > SFX_MIN_AREA_RATIO*3: score += 1
    if box_h >= SFX_BOX_HEIGHT_MIN: score += 1

    t = text_str.strip().replace(" ","").replace("\n","")
    if len(t) > 0:
        area_per_char = (box_w*box_h)/len(t)
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
) -> Tuple[Image.Image, Image.Image, int, int, list]:
    """
    FIXED v5 WORKFLOW:

    Untuk setiap teks dari Vision API:
    1. Simpan RAW bbox (koordinat Vision API asli, + small padding 15px)
    2. Gunakan raw bbox untuk DETECTION (bubble/SFX/art)
    3. Kalau dark bubble → expand ke full bubble bounds
    4. Kalau lainnya → tambah padding untuk fill area
    5. Fill:
       - Dark bubble → gradient fill (tidak pakai LaMa)
       - White bubble/dialog → gaussian blur atau LaMa
    6. Update current_np setelah fill (FIXED L4)
    """
    width, height = original.size
    total_area    = width * height
    result        = original.copy().convert("RGB")
    draw          = ImageDraw.Draw(result)
    lama_mask     = Image.new("L", (width, height), 0)
    lama_draw     = ImageDraw.Draw(lama_mask)

    # FIXED L4: current_np di-update setelah setiap fill
    current_np = img_np.copy()

    # MERGE BOXES: Gabungkan teks yang berdekatan agar diproses sebagai 1 bubble
    merged_texts = _merge_text_blocks(texts, width, height)

    inpaint_boxes = []
    sfx_count    = 0
    dialog_count = 0

    # FIXED L1/L2/L3: small padding (15px) untuk detection,
    # besar padding (PADDING=80px) hanya untuk fill
    DETECT_PAD = min(15, max(5, PADDING // 5))

    for text in merged_texts:
        # RAW bbox dari Vision API (+ small padding untuk detection)
        rx1, ry1 = max(0, text.x1 - DETECT_PAD), max(0, text.y1 - DETECT_PAD)
        rx2, ry2 = min(width, text.x2 + DETECT_PAD), min(height, text.y2 + DETECT_PAD)

        if rx1 >= rx2 or ry1 >= ry2:
            continue

        # ── STEP 1: Deteksi tipe teks/bubble ─────────────
        # Semua deteksi pakai RAW bbox (FIXED L1, L2)

        is_dark_bub  = _is_dark_bubble(current_np, rx1, ry1, rx2, ry2)
        is_white_bub = _is_white_bubble(current_np, rx1, ry1, rx2, ry2)

        # SFX check — hanya kalau bukan bubble
        if not is_dark_bub and not is_white_bub:
            if is_sfx(current_np, rx1, ry1, rx2, ry2, text.description):
                sfx_count += 1
                continue

        # Art protection — hanya kalau bukan bubble (FIXED L2)
        if not is_dark_bub and not is_white_bub:
            if is_art_text(current_np, rx1, ry1, rx2, ry2, text.description):
                continue

        dialog_count += 1

        # ── STEP 2: Tentukan fill bbox ────────────────────

        if is_dark_bub:
            # FIXED L3: expand dari RAW bbox (bukan padded)
            fx1, fy1, fx2, fy2 = _expand_bbox_to_bubble(
                current_np, rx1, ry1, rx2, ry2,
                max_expand=BUBBLE_EXPAND_DARK
            )
            fx1 = max(0, fx1); fy1 = max(0, fy1)
            fx2 = min(width, fx2); fy2 = min(height, fy2)
        elif is_white_bub:
            # Gunakan ekspansi cerdas untuk white bubble agar tidak 'makan' art
            fx1, fy1, fx2, fy2 = _expand_bbox_to_white_bubble(
                current_np, rx1, ry1, rx2, ry2
            )
        else:
            # Jika bukan bubble (teks melayang), gunakan padding kecil (25-30px)
            # Padding 80px terlalu merusak background art
            SMALL_PAD = 30
            fx1 = max(0, text.x1 - SMALL_PAD)
            fy1 = max(0, text.y1 - SMALL_PAD)
            fx2 = min(width,  text.x2 + SMALL_PAD)
            fy2 = min(height, text.y2 + SMALL_PAD)

        if fx1 >= fx2 or fy1 >= fy2:
            continue

        fw = fx2 - fx1
        fh = fy2 - fy1

        # Aspect ratio sanity check
        if fw > 0 and fh > 0:
            ratio = fw / fh
            if ratio < 0.05 or ratio > 20.0:
                continue

        # ── STEP 3: Fill ──────────────────────────────────

        if is_dark_bub:
            # Gradient fill untuk dark bubble (TIDAK pakai LaMa)
            filled = dark_bubble_gradient_fill(result, current_np, fx1, fy1, fx2, fy2)
            if filled is not None:
                result     = filled
                draw       = ImageDraw.Draw(result)
                current_np = np.array(result)  # FIXED L4
                continue
            # Fallback: LaMa (kalau luar juga gelap)
            lama_draw.rectangle([fx1, fy1, fx2, fy2], fill=255)
            continue

        # Selalu gunakan LaMa untuk dialog normal/white bubble
        lama_draw.rectangle([fx1, fy1, fx2, fy2], fill=255)
        inpaint_boxes.append((fx1, fy1, fx2, fy2))

    # Bersihkan masker dari noise kecil (misal deteksi titik/debu yang tidak perlu di-inpaint)
    if lama_mask.getbbox():
        # Hapus area putih yang luasnya kurang dari 10 piksel
        lama_mask = lama_mask.filter(ImageFilter.MinFilter(3))
        # Baru kemudian dilasi untuk menutup celah teks
        lama_mask = lama_mask.filter(ImageFilter.MaxFilter(7))
        # Perhalus tepi mask agar transisi inpainting lebih natural (anti-aliasing)
        lama_mask = lama_mask.filter(ImageFilter.GaussianBlur(radius=3))

    return result, lama_mask, sfx_count, dialog_count, inpaint_boxes