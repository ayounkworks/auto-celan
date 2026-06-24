# ============================================================
# handler.py
# RunPod Serverless + YOLO detect + LaMa inpaint
#
# Flow:
#   1. Terima gambar (base64) dari Railway
#   2. YOLO detect → bounding boxes
#   3. Build mask dari boxes
#   4. LaMa inpaint
#   5. Kembalikan gambar hasil + boxes ke Railway
# ============================================================

import base64
import os
import time
import traceback
from io import BytesIO

import runpod
import torch
import numpy as np
from PIL import Image, ImageDraw

# ── Config ────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMA_PATH  = "/app/models/big-lama.pt"
YOLO_PATH  = "/app/models/best.pt"

YOLO_CONF     = 0.25
YOLO_IOU      = 0.45
YOLO_IMGSZ    = 1024
PADDING       = 20
SLICE_HEIGHT  = 1500
SLICE_OVERLAP = 200

# ── Load models ───────────────────────────────────────────

print(f"[init] Device: {DEVICE}")
print(f"[init] Loading LaMa dari {LAMA_PATH}...")
lama_model = torch.jit.load(LAMA_PATH, map_location=DEVICE)
lama_model.eval()
lama_model.to(DEVICE)
print("[init] LaMa ready.")

print(f"[init] Loading YOLO dari {YOLO_PATH}...")
from ultralytics import YOLO
yolo_model = YOLO(YOLO_PATH)
print(f"[init] YOLO ready — task={yolo_model.task} classes={yolo_model.names}")


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
    """
    Deteksi teks dengan YOLO. Untuk gambar tinggi, slice per SLICE_HEIGHT.
    Returns list of (x1, y1, x2, y2).
    """
    w, h = image.size
    seen = set()
    all_boxes = []

    def _run_tile(tile, y_offset):
        results = yolo_model(
            tile,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            gy1 = int(y1) + y_offset
            gy2 = int(y2) + y_offset
            gx1 = int(x1)
            gx2 = int(x2)
            key = (gx1 // 5, gy1 // 5, gx2 // 5, gy2 // 5)
            if key not in seen:
                seen.add(key)
                all_boxes.append((gx1, gy1, gx2, gy2))

    if h <= int(SLICE_HEIGHT * 1.5):
        _run_tile(image, 0)
    else:
        y = 0
        while y < h:
            y2   = min(y + SLICE_HEIGHT, h)
            tile = image.crop((0, y, w, y2))
            _run_tile(tile, y)
            if y2 >= h:
                break
            y = y2 - SLICE_OVERLAP

    return all_boxes


def build_mask(image: Image.Image, boxes: list) -> Image.Image:
    """Build inpaint mask dari bounding boxes + padding."""
    w, h  = image.size
    mask  = Image.new("L", (w, h), 0)
    draw  = ImageDraw.Draw(mask)
    for (x1, y1, x2, y2) in boxes:
        px1 = max(0, x1 - PADDING)
        py1 = max(0, y1 - PADDING)
        px2 = min(w, x2 + PADDING)
        py2 = min(h, y2 + PADDING)
        draw.rectangle([px1, py1, px2, py2], fill=255)
    return mask


# ── LaMa inpainting ───────────────────────────────────────

def prepare_lama(image: Image.Image, mask: Image.Image):
    """PIL → tensor, pad ke kelipatan 8."""
    w, h = image.size
    pw = (8 - w % 8) % 8
    ph = (8 - h % 8) % 8
    if pw or ph:
        image = Image.fromarray(
            np.pad(np.array(image), ((0, ph), (0, pw), (0, 0)), mode="reflect")
        )
        mask = Image.fromarray(
            np.pad(np.array(mask), ((0, ph), (0, pw)), mode="reflect")
        )
    img_t  = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    img_t  = img_t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    mask_t = torch.from_numpy((np.array(mask) > 127).astype(np.float32))
    mask_t = mask_t.unsqueeze(0).unsqueeze(0).to(DEVICE)
    return img_t, mask_t, w, h


@torch.inference_mode()
def run_lama(image: Image.Image, mask: Image.Image) -> Image.Image:
    img_t, mask_t, orig_w, orig_h = prepare_lama(image, mask)
    out    = lama_model(img_t, mask_t)
    out_np = out[0].permute(1, 2, 0).cpu().numpy()
    out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)
    result = Image.fromarray(out_np)
    if result.size != (orig_w, orig_h):
        result = result.crop((0, 0, orig_w, orig_h))
    return result


# ── Handler ───────────────────────────────────────────────

def handler(job):
    try:
        start = time.time()
        inp   = job["input"]

        image = decode_rgb(inp["image"])
        print(f"Image: {image.size}")

        # Step 1: YOLO detect
        t1    = time.time()
        boxes = detect_boxes(image)
        print(f"YOLO: {len(boxes)} boxes ({time.time()-t1:.2f}s)")

        if not boxes:
            # Tidak ada teks — kembalikan gambar original
            print("No text detected, returning original.")
            return {"image": encode_png(image), "boxes": 0}

        # Step 2: Build mask
        mask = build_mask(image, boxes)

        # Step 3: LaMa inpaint
        t2     = time.time()
        result = run_lama(image, mask)
        print(f"LaMa: ({time.time()-t2:.2f}s)")

        elapsed = time.time() - start
        print(f"Total: {elapsed:.2f}s")

        return {
            "image": encode_png(result),
            "boxes": len(boxes),
        }

    except Exception as e:
        print("Handler error:", repr(e))
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
