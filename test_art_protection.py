# ============================================================
# test_art_protection.py
# Jalankan: python test_art_protection.py
#
# Gunakan ini untuk tune threshold art protection
# tanpa perlu run pipeline penuh.
# Taruh gambar manga di folder input/ lalu run.
# ============================================================

import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

import numpy as np
from PIL import Image, ImageDraw
from google.cloud import vision

from core.config       import GOOGLE_API_KEY, MAX_WIDTH
from core.image_processing import to_bytes
from core.art_protection   import (
    is_art_text,
    ART_PROTECT_THRESHOLD,
    _has_complex_colored_background,
    _is_gradient_bubble,
    _sample_border,
    _is_decorative_text,
)


INPUT_DIR  = os.path.join(os.path.dirname(__file__), "input")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "art_test")


def run_test(image_path: str):
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    img = Image.open(image_path).convert("RGB")
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        img   = img.resize((MAX_WIDTH, int(img.height * ratio)), Image.LANCZOS)

    img_np = np.array(img)
    w, h   = img.size

    # Deteksi teks via Vision API
    client = vision.ImageAnnotatorClient(
        client_options={"api_key": GOOGLE_API_KEY}
    )
    buf = to_bytes(img, "JPEG", 95)
    buf.seek(0)
    response = client.document_text_detection(
        image=vision.Image(content=buf.read())
    )
    texts = response.text_annotations

    if not texts:
        print("Tidak ada teks terdeteksi.")
        return

    print(f"Teks terdeteksi: {len(texts)-1} bounding box")
    print(f"Threshold proteksi: {ART_PROTECT_THRESHOLD}\n")

    # Debug tiap bounding box
    protected_boxes = []
    safe_boxes      = []

    for text in texts[1:]:
        vertices = text.bounding_poly.vertices
        xs = [v.x for v in vertices]
        ys = [v.y for v in vertices]
        x1 = max(0, min(xs) - 25)
        y1 = max(0, min(ys) - 25)
        x2 = min(w, max(xs) + 25)
        y2 = min(h, max(ys) + 25)
        text_str = text.description or ""

        # Hitung tiap sinyal secara manual
        complex_bg   = _has_complex_colored_background(img_np, x1, y1, x2, y2)
        border_px    = _sample_border(img_np, x1, y1, x2, y2, 20)
        is_bubble, _ = _is_gradient_bubble(border_px) if border_px.size > 0 else (False, 128)
        decorative   = _is_decorative_text(text_str)
        box_w        = x2 - x1
        box_h        = y2 - y1
        ratio        = box_w / box_h if box_h > 0 else 1.0
        bad_ratio    = not (0.15 <= ratio <= 12.0)

        score = 0
        if complex_bg:   score += 2
        if not is_bubble: score += 1
        if decorative:   score += 1
        if bad_ratio:    score += 1

        protected = score >= ART_PROTECT_THRESHOLD

        result = "🛡️  PROTECTED (art)" if protected else "✂️  SAFE (hapus)"
        signals = []
        if complex_bg:    signals.append("complex_bg+2")
        if not is_bubble: signals.append("no_bubble+1")
        if decorative:    signals.append("decorative+1")
        if bad_ratio:     signals.append(f"bad_ratio({ratio:.1f})+1")

        print(f"{result} | score={score} | '{text_str[:25]}' | {', '.join(signals) or 'none'}")

        if protected:
            protected_boxes.append((x1, y1, x2, y2))
        else:
            safe_boxes.append((x1, y1, x2, y2))

    # Buat visualisasi output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vis = img.copy()
    draw = ImageDraw.Draw(vis)

    for (x1, y1, x2, y2) in safe_boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)   # hijau = akan dihapus

    for (x1, y1, x2, y2) in protected_boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(255, 50, 50), width=3)  # merah = dilindungi

    out_name = "debug_" + os.path.basename(image_path)
    out_path = os.path.join(OUTPUT_DIR, out_name)
    vis.save(out_path)

    print(f"\n📊 Ringkasan:")
    print(f"   ✂️  Akan dihapus : {len(safe_boxes)} box  (kotak hijau)")
    print(f"   🛡️  Dilindungi   : {len(protected_boxes)} box  (kotak merah)")
    print(f"   📁 Visualisasi  : {out_path}")
    print()
    print("Kalau terlalu banyak yang dilindungi → turunkan ART_PROTECT_THRESHOLD di art_protection.py")
    print("Kalau art masih terhapus             → naikkan ART_PROTECT_THRESHOLD")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Mode: python test_art_protection.py path/to/image.jpg
        for path in sys.argv[1:]:
            if os.path.exists(path):
                run_test(path)
            else:
                print(f"File tidak ditemukan: {path}")
    else:
        # Mode: scan semua gambar di input/
        exts  = (".jpg", ".jpeg", ".png", ".webp")
        files = [
            os.path.join(INPUT_DIR, f)
            for f in sorted(os.listdir(INPUT_DIR))
            if f.lower().endswith(exts)
        ]
        if not files:
            print(f"Tidak ada gambar di {INPUT_DIR}")
            sys.exit(1)
        for f in files:
            run_test(f)
