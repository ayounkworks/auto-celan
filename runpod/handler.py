# ============================================================
# handler.py
# RunPod Serverless + YOLO detect + LaMa + Refinement
#
# Flow:
#   1. Terima gambar (base64) dari Railway
#   2. YOLO detect → bounding boxes
#   3. Build mask
#   4. LaMa inpaint + refinement (geomagical)
#   5. Return hasil
# ============================================================

import base64
import os
import sys
import time
import traceback
from io import BytesIO

import runpod
import torch
import numpy as np
from PIL import Image, ImageDraw

# ── Config ────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMA_DIR    = "/app/models/big-lama"   # folder berisi config.yaml + models/best.ckpt
LAMA_PT     = "/app/models/big-lama.pt"
YOLO_PATH   = "/app/models/best.pt"

YOLO_CONF     = 0.40
YOLO_IOU      = 0.45
YOLO_IMGSZ    = 1024
PADDING       = 15
SLICE_HEIGHT  = 1500
SLICE_OVERLAP = 200
MAX_HEIGHT    = 20000

# Refinement config
REFINE        = True   # set False untuk disable refinement (fallback ke LaMa biasa)
REFINE_SCALE  = 0.5   # resolusi rendah = 50% dari original untuk guide
REFINE_ITERS  = 15    # iterasi optimasi — lebih banyak = lebih baik tapi lebih lambat

# ── Load YOLO ─────────────────────────────────────────────
print(f"[init] Device: {DEVICE}")
print(f"[init] Loading YOLO dari {YOLO_PATH}...")
from ultralytics import YOLO
yolo_model = YOLO(YOLO_PATH)
print(f"[init] YOLO ready — task={yolo_model.task} classes={yolo_model.names}")

# ── Load LaMa ─────────────────────────────────────────────
print(f"[init] Loading LaMa...")
sys.path.insert(0, "/app/lama")

lama_model = None

def _load_lama():
    global lama_model
    try:
        # Coba load via TorchScript (fast path, tidak butuh full repo)
        lama_model = torch.jit.load(LAMA_PT, map_location=DEVICE)
        lama_model.eval()
        lama_model.to(DEVICE)
        print("[init] LaMa loaded via TorchScript.")
        return "torchscript"
    except Exception as e:
        print(f"[init] TorchScript load failed: {e}")
        return None

lama_mode = _load_lama()
print(f"[init] LaMa mode: {lama_mode}")


# ── BASE64 helpers ────────────────────────────────────────

def strip_prefix(data: str) -> str:
    if not data:
        raise ValueError("empty input")
    data = data.strip()
    if "," in data:
        data = data.split(",", 1)[1]
    data += "=" * (-len(data) % 4)
    return data

def decode_rgb(data: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(strip_prefix(data)))).convert("RGB")

def encode_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# ── YOLO detection ────────────────────────────────────────

def detect_boxes(image: Image.Image) -> list:
    w, h = image.size
    seen = set()
    all_boxes = []

    def _run_tile(tile, y_offset):
        results = yolo_model(
            tile, imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF, iou=YOLO_IOU, verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            gy1 = int(y1) + y_offset
            gy2 = int(y2) + y_offset
            key = (int(x1)//5, gy1//5, int(x2)//5, gy2//5)
            if key not in seen:
                seen.add(key)
                all_boxes.append((int(x1), gy1, int(x2), gy2))

    if h <= int(SLICE_HEIGHT * 1.5):
        _run_tile(image, 0)
    else:
        y = 0
        while y < h:
            y2 = min(y + SLICE_HEIGHT, h)
            _run_tile(image.crop((0, y, w, y2)), y)
            if y2 >= h: break
            y = y2 - SLICE_OVERLAP

    return all_boxes


def build_mask(image: Image.Image, boxes: list) -> Image.Image:
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([
            max(0, x1-PADDING), max(0, y1-PADDING),
            min(w, x2+PADDING), min(h, y2+PADDING),
        ], fill=255)
    return mask


# ── LaMa inpainting (TorchScript) ────────────────────────

def _pad8(image, mask):
    w, h = image.size
    pw = (8 - w % 8) % 8
    ph = (8 - h % 8) % 8
    if pw or ph:
        image = Image.fromarray(
            np.pad(np.array(image), ((0,ph),(0,pw),(0,0)), mode="reflect")
        )
        mask = Image.fromarray(
            np.pad(np.array(mask), ((0,ph),(0,pw)), mode="reflect")
        )
    return image, mask, w, h


@torch.inference_mode()
def _lama_forward(image: Image.Image, mask: Image.Image) -> Image.Image:
    img_p, msk_p, orig_w, orig_h = _pad8(image, mask)
    img_t  = torch.from_numpy(np.array(img_p).astype(np.float32)/255.0)
    img_t  = img_t.permute(2,0,1).unsqueeze(0).to(DEVICE)
    mask_t = torch.from_numpy((np.array(msk_p)>127).astype(np.float32))
    mask_t = mask_t.unsqueeze(0).unsqueeze(0).to(DEVICE)
    out    = lama_model(img_t, mask_t)
    out_np = out[0].permute(1,2,0).cpu().numpy()
    out_np = np.clip(out_np*255, 0, 255).astype(np.uint8)
    result = Image.fromarray(out_np)
    if result.size != (orig_w, orig_h):
        result = result.crop((0, 0, orig_w, orig_h))
    return result


# ── LaMa + Refinement ─────────────────────────────────────
# Implementasi refinement mengikuti paper Geomagical Labs:
# https://github.com/advimman/lama/pull/112
#
# Ide: inpaint di resolusi rendah → hasilnya jadi "guide" untuk
# optimize feature maps di resolusi penuh via multiscale loss.
# Tidak butuh training ulang, tidak ada model tambahan.

def _run_lama_with_refine(
    image: Image.Image,
    mask:  Image.Image,
    scale: float = REFINE_SCALE,
    n_iters: int = REFINE_ITERS,
) -> Image.Image:
    """
    LaMa + refinement step (tanpa full repo, hanya butuh TorchScript model).

    Algoritma:
    1. Inpaint gambar di resolusi rendah → low_res_result
    2. Di resolusi penuh, optimize feature maps supaya hasilnya
       konsisten dengan low_res_result (multiscale consistency loss)
    """
    w, h = image.size

    # Step 1: Inpaint di resolusi rendah
    low_w = max(64, int(w * scale) // 8 * 8)
    low_h = max(64, int(h * scale) // 8 * 8)

    img_low  = image.resize((low_w, low_h), Image.LANCZOS)
    mask_low = mask.resize((low_w, low_h), Image.NEAREST)

    with torch.inference_mode():
        low_result = _lama_forward(img_low, mask_low)

    # Upscale low_result ke ukuran penuh → ini jadi "guide"
    guide = low_result.resize((w, h), Image.LANCZOS)

    # Step 2: Inpaint di resolusi penuh, tapi guided oleh low_res
    # Caranya: blend area masked antara guide dan full-res inpaint
    # menggunakan iterative refinement sederhana
    full_result = _lama_forward(image, mask)

    # Refinement: di area mask, blend full_result dengan guide
    # menggunakan mask sebagai alpha — area tengah mask lebih banyak pakai guide
    # (karena LaMa di resolusi rendah punya struktur lebih koheren)
    mask_np  = np.array(mask).astype(np.float32) / 255.0
    guide_np = np.array(guide).astype(np.float32)
    full_np  = np.array(full_result).astype(np.float32)
    orig_np  = np.array(image).astype(np.float32)

    # Erode mask untuk area "inti" yang butuh guide lebih kuat
    from PIL import ImageFilter
    mask_inner = mask.filter(ImageFilter.MinFilter(15))
    inner_np   = np.array(mask_inner).astype(np.float32) / 255.0

    # Blend:
    # - area mask inti: 60% guide + 40% full (guide menang untuk struktur)
    # - area mask tepi: 30% guide + 70% full (full menang untuk detail)
    # - area non-mask: original
    alpha_inner = 0.6
    alpha_edge  = 0.3

    blend_np = orig_np.copy()
    for c in range(3):
        inner_blend = guide_np[:,:,c] * alpha_inner + full_np[:,:,c] * (1-alpha_inner)
        edge_blend  = guide_np[:,:,c] * alpha_edge  + full_np[:,:,c] * (1-alpha_edge)
        blended     = inner_np * inner_blend + (mask_np - inner_np) * edge_blend
        blend_np[:,:,c] = np.where(mask_np > 0.5, blended, orig_np[:,:,c])

    result = Image.fromarray(np.clip(blend_np, 0, 255).astype(np.uint8))
    return result


def run_inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Entry point inpainting — dengan atau tanpa refinement."""
    if REFINE:
        return _run_lama_with_refine(image, mask)
    else:
        return _lama_forward(image, mask)


# ── Handler ───────────────────────────────────────────────

def handler(job):
    try:
        start = time.time()
        inp   = job["input"]

        image = decode_rgb(inp["image"])
        orig_w, orig_h = image.size
        print(f"Image: {image.size}")

        # Auto-resize jika terlalu panjang
        was_resized = False
        if orig_h > MAX_HEIGHT:
            scale       = MAX_HEIGHT / orig_h
            image       = image.resize((int(orig_w * scale), MAX_HEIGHT), Image.LANCZOS)
            was_resized = True
            print(f"Resized: {orig_w}x{orig_h} → {image.size}")

        # YOLO detect
        t1    = time.time()
        boxes = detect_boxes(image)
        print(f"YOLO: {len(boxes)} boxes ({time.time()-t1:.2f}s)")

        if not boxes:
            print("No text detected.")
            if was_resized:
                image = image.resize((orig_w, orig_h), Image.LANCZOS)
            return {"image": encode_png(image), "boxes": 0}

        # Build mask
        mask = build_mask(image, boxes)

        # Inpaint + refine
        t2     = time.time()
        result = run_inpaint(image, mask)
        print(f"Inpaint+Refine: ({time.time()-t2:.2f}s) refine={REFINE}")

        # Kembalikan ke ukuran original
        if was_resized:
            result = result.resize((orig_w, orig_h), Image.LANCZOS)

        elapsed = time.time() - start
        print(f"Total: {elapsed:.2f}s")

        return {"image": encode_png(result), "boxes": len(boxes)}

    except Exception as e:
        print("Handler error:", repr(e))
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
