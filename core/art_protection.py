# ============================================================
# core/art_protection.py
# Art Protection — filter teks yang merupakan bagian dari art,
# bukan dialog/narasi yang perlu dihapus.
#
# Masalah yang diselesaikan:
#   Manga berwarna dengan bubble transparan/gradient sering punya
#   background kompleks di balik teks. Vision API tidak bisa
#   bedakan "teks dialog" vs "teks bagian dari gambar" (kaos, sign,
#   tato, nama toko, logo, dll). Modul ini menambahkan 5 sinyal
#   untuk memutuskan apakah sebuah bounding box aman dihapus atau
#   harus dilindungi.
#
# Returns: is_art_text(img_np, x1, y1, x2, y2, text_str) -> bool
#   True  = lindungi, jangan hapus
#   False = aman, boleh hapus
# ============================================================

import numpy as np
from typing import Tuple


# ── Konstanta ─────────────────────────────────────────────

# Minimum confidence score untuk dianggap "aman dihapus"
# Semakin tinggi = semakin ketat perlindungan art
ART_PROTECT_THRESHOLD = 4

# Warna bubble yang umum di manga berwarna
# (putih bersih, hitam solid, gradient hitam-putih)
BUBBLE_WHITE_MIN    = 210   # pixel dianggap putih
BUBBLE_BLACK_MAX    = 45    # pixel dianggap hitam
BUBBLE_GRAY_MAX_STD = 30.0  # std rendah = background polos/bubble

# Ukuran border sample untuk analisis sekitar teks
ART_BORDER_PAD = 20

# Rasio minimum pixel terang di sekitar teks
# untuk dianggap bubble (bukan art)
BUBBLE_BRIGHT_RATIO = 0.55

# Minimum area ratio untuk aktifkan pengecekan
# (teks sangat kecil tidak perlu dicek, biasanya dialog biasa)
ART_MIN_AREA_RATIO  = 0.0003

# Gradient detection threshold
# Bubble transparan/gradient biasanya punya transisi warna halus
GRADIENT_STD_LOW    = 18.0   # terlalu polos = solid bubble
GRADIENT_STD_HIGH   = 75.0   # terlalu kompleks = mungkin art


# ── Helpers ───────────────────────────────────────────────

def _sample_border(img_np: np.ndarray,
                   x1: int, y1: int, x2: int, y2: int,
                   pad: int) -> np.ndarray:
    """Ambil pixel di sekitar bounding box (bukan di dalamnya)."""
    h, w  = img_np.shape[:2]
    bx1   = max(0, x1 - pad)
    by1   = max(0, y1 - pad)
    bx2   = min(w, x2 + pad)
    by2   = min(h, y2 + pad)

    strips = []
    # atas
    if by1 < y1:
        strips.append(img_np[by1:y1, bx1:bx2])
    # bawah
    if y2 < by2:
        strips.append(img_np[y2:by2, bx1:bx2])
    # kiri
    if bx1 < x1:
        strips.append(img_np[y1:y2, bx1:x1])
    # kanan
    if x2 < bx2:
        strips.append(img_np[y1:y2, x2:bx2])

    if not strips:
        return np.array([])

    all_px = np.concatenate([s.reshape(-1, s.shape[2]) for s in strips
                             if s.size > 0 and s.ndim == 3], axis=0)
    return all_px


def _is_gradient_bubble(border_px: np.ndarray) -> Tuple[bool, float]:
    """
    Deteksi bubble transparan/gradient hitam.
    Bubble gradient biasanya: pixel gelap + std sedang (tidak polos, tidak complex).
    Returns (is_bubble, mean_brightness)
    """
    if border_px.size == 0:
        return False, 128.0

    gray = border_px.mean(axis=1)
    mean = float(gray.mean())
    std  = float(gray.std())

    # Bubble solid putih
    if mean > BUBBLE_WHITE_MIN and std < BUBBLE_GRAY_MAX_STD:
        return True, mean

    # FIX Bug1: Bubble solid hitam — hapus syarat std
    # Dark bubble dengan teks putih di dalamnya punya std tinggi (kontras putih-hitam)
    # tapi tetap harus dianggap bubble, bukan art
    if mean < BUBBLE_BLACK_MAX:
        return True, mean

    # Bubble gradient (transparan ke hitam/putih)
    # Ciri: mean di range gelap-sedang, std tidak terlalu tinggi
    if mean < 160 and GRADIENT_STD_LOW < std < GRADIENT_STD_HIGH:
        return True, mean

    return False, mean


def _has_complex_colored_background(img_np: np.ndarray,
                                    x1: int, y1: int,
                                    x2: int, y2: int) -> bool:
    """
    Cek apakah area di dalam bounding box punya background
    yang kompleks dan berwarna — sinyal kuat bahwa ini art, bukan bubble.
    """
    h, w   = img_np.shape[:2]
    pad    = 4
    ix1    = min(x1 + pad, x2 - 1)
    iy1    = min(y1 + pad, y2 - 1)
    ix2    = max(ix1 + 1, x2 - pad)
    iy2    = max(iy1 + 1, y2 - pad)

    region = img_np[iy1:iy2, ix1:ix2]
    if region.size == 0:
        return False

    # Pisahkan channel R, G, B
    r_std = float(np.std(region[:, :, 0]))
    g_std = float(np.std(region[:, :, 1]))
    b_std = float(np.std(region[:, :, 2]))

    # Variance per channel tinggi = background kompleks/berwarna
    avg_std = (r_std + g_std + b_std) / 3

    # Juga cek apakah warna beragam (bukan grayscale)
    rg_diff = float(np.mean(np.abs(
        region[:, :, 0].astype(float) - region[:, :, 1].astype(float)
    )))
    rb_diff = float(np.mean(np.abs(
        region[:, :, 0].astype(float) - region[:, :, 2].astype(float)
    )))
    color_diversity = (rg_diff + rb_diff) / 2

    # Background kompleks berwarna: std tinggi DAN ada variasi warna
    return avg_std > 70.0 and color_diversity > 30.0


def _is_decorative_text(text_str: str) -> bool:
    """
    Teks dekoratif yang biasanya bagian dari art:
    - Teks sangat pendek (1-2 karakter) — bisa nama, inisial di art
    - Mixed script yang aneh (bukan dialog biasa)
    - Angka saja — bisa bagian dari art/environment
    """
    t = text_str.strip().replace(" ", "").replace("\n", "")
    if not t:
        return False

    # Angka saja (1-3 digit) — bisa nomor halaman, tapi juga bisa art
    # Hanya flag kalau sangat pendek
    if t.isdigit() and len(t) <= 2:
        return True

    # Single karakter non-alfanumerik
    if len(t) == 1 and not t.isalnum():
        return True

    return False


def _aspect_ratio_ok(box_w: int, box_h: int) -> bool:
    """
    Dialog bubble biasanya punya aspect ratio wajar untuk teks:
    lebar > tinggi (teks horizontal) atau hampir kotak.
    Teks yang sangat tinggi-sempit atau sangat lebar-pendek
    lebih mungkin bagian dari art/environment.
    """
    if box_h == 0:
        return True
    ratio = box_w / box_h
    # Terlalu sempit vertikal (ratio < 0.15) atau
    # terlalu lebar horizontal (ratio > 12) → mencurigakan
    return 0.15 <= ratio <= 12.0


# ── Main Function ─────────────────────────────────────────

def is_art_text(
    img_np:   np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    text_str: str = "",
) -> bool:
    """
    Sistem voting untuk deteksi apakah teks ini bagian dari art.

    Scoring:
      +2 → background dalam bounding box kompleks berwarna (sinyal kuat)
      +1 → border sekitar teks BUKAN bubble (tidak putih/hitam/gradient)
      +1 → teks dekoratif (angka pendek, karakter tunggal)
      +1 → aspect ratio aneh untuk dialog
      +1 → area sangat kecil di background kompleks

    Total >= ART_PROTECT_THRESHOLD → lindungi (return True)
    """
    h, w       = img_np.shape[:2]
    total_px   = h * w
    box_w      = x2 - x1
    box_h      = y2 - y1
    box_area   = box_w * box_h

    # Terlalu kecil untuk dicek → anggap aman (dialog kecil biasa)
    if box_area / total_px < ART_MIN_AREA_RATIO:
        return False

    score = 0

    # ── FIX Bug1: Dark bubble early-exit ──────────────────────────
    # Kalau border sekitar box sangat gelap (mean < 50) → ini dark bubble
    # Dark bubble tidak perlu voting — langsung aman dihapus (return False)
    border_px_check = _sample_border(img_np, x1, y1, x2, y2, ART_BORDER_PAD)
    if border_px_check.size > 0:
        border_gray = border_px_check.mean(axis=1)
        border_mean = float(border_gray.mean())
        # Inside box: cek apakah dominan gelap (dark bubble interior)
        pad = 4
        ix1c = min(x1 + pad, x2 - 1)
        iy1c = min(y1 + pad, y2 - 1)
        ix2c = max(ix1c + 1, x2 - pad)
        iy2c = max(iy1c + 1, y2 - pad)
        interior = img_np[iy1c:iy2c, ix1c:ix2c]
        interior_mean = float(interior.mean()) if interior.size > 0 else 128.0
        # Dark bubble: interior gelap (< 80) ATAU border gelap (< 50)
        if interior_mean < 80 or border_mean < 50:
            return False  # dark bubble → jangan proteksi, biarkan dihapus

    # ── Sinyal 1 (bobot 2): background dalam box sangat berwarna ──
    if _has_complex_colored_background(img_np, x1, y1, x2, y2):
        score += 1

    # ── Sinyal 2 (bobot 1): border sekitar box bukan bubble ──
    border_px = _sample_border(img_np, x1, y1, x2, y2, ART_BORDER_PAD)
    if border_px.size > 0:
        is_bubble, _ = _is_gradient_bubble(border_px)
        if not is_bubble:
            score += 1

    # ── Sinyal 3 (bobot 1): teks dekoratif ──
    if _is_decorative_text(text_str):
        score += 1

    # ── Sinyal 4 (bobot 1): aspect ratio tidak wajar untuk dialog ──
    if not _aspect_ratio_ok(box_w, box_h):
        score += 1

    # ── Sinyal 5 (bobot 1): area kecil di tengah background kompleks ──
    # Teks kecil yang nempel di art (kaos, sign kecil)
    if (box_area / total_px < 0.002
            and score >= 1):  # sudah ada sinyal lain
        score += 1

    protected = score >= ART_PROTECT_THRESHOLD

    if protected:
        print(f"  [art_protect] Skip '{text_str[:20]}' score={score} "
              f"box=({x1},{y1},{x2},{y2})")

    return protected