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
GRADIENT_STD_HIGH   = 72.0   # lebih toleran untuk gradient bubble


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


def _has_complex_colored_background(img_np: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Cek apakah area di dalam bounding box punya background
    yang kompleks dan berwarna — sinyal kuat bahwa ini art, bukan bubble.
    """
    roi = img_np[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    
    # Cek mean brightness dulu - kalau gelap sekali (dark bubble), 
    # jangan langsung dianggap complex art background
    mean_brightness = float(roi.mean())
    
    stds = roi.std(axis=(0,1))
    avg_std = float(stds.mean())
    color_diversity = float(stds.max() - stds.min())
    
    # Dark circular bubble memiliki mean rendah tapi std juga rendah
    # (background gelap solid dengan teks putih)
    # → threshold lebih ketat
    if mean_brightness < 80:
        # Dark region: perlu std SANGAT tinggi untuk dianggap art
        return avg_std > 85.0 and color_diversity > 40.0
    
    # Normal: naik threshold dari 55→70 dan 20→28
    return avg_std > 70.0 and color_diversity > 28.0


def _is_dark_bubble_region(img_np: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Deteksi apakah ini region dark circular bubble (seperti di manhwa Korea).
    Dark bubble = background gelap (#000-#333) dengan teks putih/terang di tengah.
    BUKAN art - ini dialog/caption bubble tipe gelap.
    """
    roi = img_np[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    
    mean_val = float(roi.mean())
    # Background gelap
    if mean_val > 120:
        return False
    
    # Cek apakah ada pixels terang di tengah (teks putih)
    roi_h, roi_w = roi.shape[:2]
    center = roi[roi_h//4:3*roi_h//4, roi_w//4:3*roi_w//4]
    if center.size == 0:
        return False
    
    bright_ratio = float((center > 180).mean())
    # Ada teks putih di tengah dark background = dark bubble
    return bright_ratio > 0.05


def _is_ornate_frame_bubble(img_np: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Deteksi bubble dengan ornate/decorative frame (pink/gold border).
    Ciri: interior sangat putih, border berwarna di tepi.
    Ini tetap harus dihapus (dialog bubble, bukan art).
    """
    roi = img_np[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 40 or roi_w < 40:
        return False
    
    # Interior (80% tengah) harus sangat putih
    margin_y = roi_h // 8
    margin_x = roi_w // 8
    interior = roi[margin_y:roi_h-margin_y, margin_x:roi_w-margin_x]
    
    if interior.size == 0:
        return False
    
    interior_white = float((interior > 235).all(axis=2).mean())
    
    # Interior putih ≥ 70% = ornate frame bubble
    return interior_white > 0.70


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
      +1 → background dalam bounding box kompleks berwarna
      +1 → border sekitar teks BUKAN bubble (tidak putih/hitam/gradient)
      +1 → teks dekoratif (angka pendek, karakter tunggal)
      +1 → aspect ratio aneh untuk dialog
      +2 → teks sangat besar (kemungkinan judul/art decoration)
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

    # ── Early exit: jelas bukan art ──

    # 1. Dark bubble (manhwa dark circular dialog)
    if _is_dark_bubble_region(img_np, x1, y1, x2, y2):
        return False  # Ini dialog, bukan art

    # 2. Ornate frame bubble (interior putih dengan fancy border)
    if _is_ornate_frame_bubble(img_np, x1, y1, x2, y2):
        return False  # Ini dialog bubble, bukan art

    # 3. Teks di area pinggir (side panel) dengan background complex
    #    Kemungkinan besar dialog panel, bukan art text
    img_w = img_np.shape[1]
    is_side_text = (x2 < img_w * 0.30) or (x1 > img_w * 0.70)

    # ── Sinyal 1: background dalam box sangat berwarna ──
    if _has_complex_colored_background(img_np, x1, y1, x2, y2):
        # Guard: kalau ini dark bubble atau side panel, jangan tambah score
        if not _is_dark_bubble_region(img_np, x1, y1, x2, y2) and not is_side_text:
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

    # ── Sinyal baru: teks sangat besar = kemungkinan art title/decoration ──
    if box_area / total_px > 0.06:  # > 6% area gambar
        score += 2  # Kemungkinan besar title/judul dekoratif

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