# ============================================================
# handler.py
# RunPod Serverless + LaMa (TorchScript, big-lama.pt)
#
# Model: big-lama.pt dari enesmsahin/simple-lama-inpainting
# Format: torch.jit.load (TorchScript) — tidak butuh saicinpainting
# ============================================================

import base64
import os
import time
import traceback
from io import BytesIO

import runpod
import torch
import numpy as np
from PIL import Image

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/app/models/big-lama.pt"

print(f"[init] Device: {DEVICE}")
print(f"[init] Loading {MODEL_PATH}...")

model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()
model.to(DEVICE)

print("[init] Model ready.")


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

def decode_mask(data: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(strip_prefix(data)))).convert("L")

def encode_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Preprocessing ─────────────────────────────────────────

def prepare(image: Image.Image, mask: Image.Image, device):
    """PIL → tensor siap masuk LaMa TorchScript."""
    # Pad ke kelipatan 8
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
    img_t  = img_t.permute(2, 0, 1).unsqueeze(0).to(device)   # (1,3,H,W)

    mask_t = torch.from_numpy((np.array(mask) > 127).astype(np.float32))
    mask_t = mask_t.unsqueeze(0).unsqueeze(0).to(device)       # (1,1,H,W)

    return img_t, mask_t, w, h


# ── Inference ─────────────────────────────────────────────

@torch.inference_mode()
def run_lama(image: Image.Image, mask: Image.Image) -> Image.Image:
    img_t, mask_t, orig_w, orig_h = prepare(image, mask, DEVICE)

    out = model(img_t, mask_t)

    # (1,3,H,W) → (H,W,3)
    out_np = out[0].permute(1, 2, 0).cpu().numpy()
    out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)

    result = Image.fromarray(out_np)

    # Crop kembali ke ukuran original jika ada padding
    if result.size != (orig_w, orig_h):
        result = result.crop((0, 0, orig_w, orig_h))

    return result


# ── Handler ───────────────────────────────────────────────

def handler(job):
    try:
        start = time.time()
        inp   = job["input"]

        image = decode_rgb(inp["image"])
        mask  = decode_mask(inp["mask"])

        print(f"Image: {image.size}  Mask: {mask.size}")

        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)

        result  = run_lama(image, mask)
        elapsed = time.time() - start

        print(f"Done in {elapsed:.2f}s.")
        return {"image": encode_png(result)}

    except Exception as e:
        print("Handler error:", repr(e))
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})