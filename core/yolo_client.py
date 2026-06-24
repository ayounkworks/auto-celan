# ============================================================
# core/yolo_client.py
# YOLOv11 inference — supports both detect and segment tasks
#
# Input : PIL Image (any size)
# Output: list of _Ann objects compatible dengan _merge_text_blocks()
#
# Model: best.pt  (nc=1, task=detect, class=text_region)
# ============================================================

import os
import numpy as np
from PIL import Image, ImageDraw
from typing import Optional

_model = None
_model_path: Optional[str] = None

YOLO_CONF     = 0.25
YOLO_IOU      = 0.45
YOLO_IMGSZ    = 1024

SLICE_HEIGHT  = 1500
SLICE_OVERLAP = 200


def load_model(model_path: str):
    global _model, _model_path
    if _model is None or _model_path != model_path:
        from ultralytics import YOLO
        print(f"[yolo_client] Loading model: {model_path}")
        _model = YOLO(model_path)
        _model_path = model_path
        print(f"[yolo_client] Model loaded — task={_model.task} classes={_model.names}")
    return _model


# ── Fake annotation objects ────────────────────────────────
# Compatible dengan MergedText / _merge_text_blocks() di image_processing.py

class _V:
    __slots__ = ("x", "y")
    def __init__(self, x, y): self.x = x; self.y = y

class _BP:
    __slots__ = ("vertices",)
    def __init__(self, verts): self.vertices = verts

class _Ann:
    __slots__ = ("description", "bounding_poly")
    def __init__(self, desc, bp):
        self.description   = desc
        self.bounding_poly = bp


def _bbox_to_ann(x1, y1, x2, y2, label="text") -> _Ann:
    verts = [_V(x1, y1), _V(x2, y1), _V(x2, y2), _V(x1, y2)]
    return _Ann(label, _BP(verts))


# ── Core inference ─────────────────────────────────────────

def _run_yolo_on_tile(model, tile: Image.Image, y_offset: int, seen: set) -> list:
    """
    Jalankan YOLO pada satu tile.
    Support task=detect (bbox) maupun task=segment (polygon).
    Kembalikan list _Ann dengan koordinat sudah di-offset ke gambar penuh.
    """
    results = model(
        tile,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        verbose=False,
    )
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return []

    anns  = []
    boxes = r.boxes.xyxy.cpu().numpy()  # (N, 4) — x1 y1 x2 y2

    for box in boxes:
        x1, y1, x2, y2 = box
        gy1 = float(y1) + y_offset
        gy2 = float(y2) + y_offset
        gx1 = float(x1)
        gx2 = float(x2)

        key = (int(gx1 // 5), int(gy1 // 5), int(gx2 // 5), int(gy2 // 5))
        if key in seen:
            continue
        seen.add(key)

        anns.append(_bbox_to_ann(int(gx1), int(gy1), int(gx2), int(gy2)))

    return anns


def detect_text(image: Image.Image, model_path: str) -> list:
    """
    Deteksi teks pada gambar menggunakan YOLO.
    Untuk gambar tinggi (webtoon), slice per SLICE_HEIGHT.

    Returns list _Ann compatible dengan _merge_text_blocks().
    Index 0 adalah dummy full-image annotation (seperti Vision API).
    Returns [] jika tidak ada deteksi.
    """
    model = load_model(model_path)
    w, h  = image.size

    seen     = set()
    all_anns: list[_Ann] = []

    if h <= int(SLICE_HEIGHT * 1.5):
        all_anns = _run_yolo_on_tile(model, image, y_offset=0, seen=seen)
    else:
        y = 0
        while y < h:
            y2   = min(y + SLICE_HEIGHT, h)
            tile = image.crop((0, y, w, y2))
            tile_anns = _run_yolo_on_tile(model, tile, y_offset=y, seen=seen)
            all_anns.extend(tile_anns)
            if y2 >= h:
                break
            y = y2 - SLICE_OVERLAP

    if not all_anns:
        return []

    # Dummy full-image annotation di index 0 — kompatibel dengan _merge_text_blocks
    dummy = _bbox_to_ann(0, 0, w, h, label="__full__")
    return [dummy] + all_anns


# ── Mask builder (untuk debug saja) ───────────────────────

def build_mask_from_yolo(
    image: Image.Image,
    model_path: str,
    padding: int = 20,
) -> Image.Image:
    """
    Bangun lama_mask langsung dari YOLO output.
    Untuk task=segment: pakai polygon mask.
    Untuk task=detect: pakai bounding box.
    Berguna untuk debug/visual inspection, tidak dipakai di pipeline utama.
    """
    model = load_model(model_path)
    w, h  = image.size
    mask  = Image.new("L", (w, h), 0)
    draw  = ImageDraw.Draw(mask)

    results = model(
        image,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        verbose=False,
        retina_masks=(model.task == "segment"),
    )
    r = results[0]

    if model.task == "segment" and r.masks is not None:
        for poly in r.masks.xy:
            if len(poly) < 3:
                continue
            pts = [(int(x), int(y)) for x, y in poly]
            draw.polygon(pts, fill=255)
    elif r.boxes is not None:
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = [int(v) for v in box]
            draw.rectangle([x1, y1, x2, y2], fill=255)

    if padding > 0:
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.MaxFilter(padding * 2 + 1))

    return mask