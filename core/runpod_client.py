# ============================================================
# core/runpod_client.py
# FIXED v2:
# - BUG2: Guard img is not None sebelum img.size (line 160, 234)
# ============================================================

import asyncio
import base64
import time
import math
from io import BytesIO
from typing import Optional

import aiohttp
from PIL import Image

from core.config import (
    RUNPOD_API_KEY,
    RUNPOD_ENDPOINT_ID,
)

runpod_sem: Optional[asyncio.Semaphore] = None

RUNSYNC_TIMEOUT = 55
POLL_INTERVAL   = 0.5
POLL_MAX        = 160

LAMA_MIN_SIZE = 128
TILE_SIZE     = 512


def _prepare_slice(image, mask):
    """Pastikan slice kelipatan 8 dengan padding (bukan resizing)."""
    orig_w, orig_h = image.size
    tw = math.ceil(orig_w / 8) * 8
    th = math.ceil(orig_h / 8) * 8

    padded_img = Image.new("RGB", (tw, th), (255, 255, 255))
    padded_img.paste(image, (0, 0))
    
    padded_mask = Image.new("L", (tw, th), 0)
    padded_mask.paste(mask, (0, 0))
    
    return padded_img, padded_mask, orig_w, orig_h


def normalize_b64(data: str):
    data = data.strip()
    if "," in data:
        data = data.split(",", 1)[1].strip()
    data += "=" * (-len(data) % 4)
    return data


def extract_output(output):
    if isinstance(output, dict):
        for key in ("image", "image_url", "message"):
            if key in output:
                return output[key]
        if "images" in output:
            images = output["images"]
            if images:
                first = images[0]
                if isinstance(first, dict):
                    return first.get("image") or first.get("url")
    if isinstance(output, list) and output:
        return extract_output(output[0])
    if isinstance(output, str):
        return output
    return None


async def decode_image_async(output, http_session=None):
    output = extract_output(output)
    if not output:
        return None

    if isinstance(output, str) and output.startswith("http"):
        session = http_session or aiohttp.ClientSession()
        try:
            async with session.get(output, timeout=aiohttp.ClientTimeout(total=60)) as r:
                if r.status != 200:
                    return None
                content = await r.read()
                return Image.open(BytesIO(content)).convert("RGB")
        finally:
            if http_session is None:
                await session.close()

    if isinstance(output, str):
        try:
            b64 = normalize_b64(output)
            return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception as e:
            print(f"[RunPod decode error] {repr(e)}")
            return None
    return None


def _build_payload(image, mask) -> dict:
    img_buf  = BytesIO()
    mask_buf = BytesIO()
    image.save(img_buf,  format="JPEG", quality=95)
    mask.save(mask_buf,  format="PNG")
    return {
        "input": {
            "image": "data:image/jpeg;base64," + base64.b64encode(img_buf.getvalue()).decode(),
            "mask":  "data:image/png;base64,"  + base64.b64encode(mask_buf.getvalue()).decode(),
        }
    }


async def _run_runsync(image, mask, label="", http_session=None) -> Optional[Image.Image]:
    img_send, mask_send, sw, sh = _prepare_slice(image, mask)
    payload = _build_payload(img_send, mask_send)
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type":  "application/json",
    }
    url   = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"
    start = time.time()

    try:
        async with http_session.post(
            url, headers=headers, json=payload,
            timeout=aiohttp.ClientTimeout(total=RUNSYNC_TIMEOUT),
        ) as r:
            if r.status not in (200, 201):
                text = await r.text()
                print(f"[{label}] runsync HTTP {r.status}: {text[:200]}")
                return None
            data = await r.json()

        status = data.get("status")
        if status == "COMPLETED":
            img     = await decode_image_async(data.get("output"), http_session=http_session)
            elapsed = time.time() - start
            # FIXED BUG2: guard img is not None sebelum .size
            if img is None:
                print(f"[{label}] runsync: output tidak bisa di-decode ({elapsed:.2f}s)")
                return None
            if img.size != (sw, sh):
                img = img.crop((0, 0, sw, sh))
            print(f"[{label}] runsync selesai dalam {elapsed:.2f}s")
            return img
        elif status == "FAILED":
            print(f"[{label}] runsync FAILED: {data.get('error')} ({time.time()-start:.2f}s)")
            return None
        else:
            print(f"[{label}] runsync status={status}, fallback ke polling")
            return None

    except asyncio.TimeoutError:
        print(f"[{label}] runsync timeout setelah {RUNSYNC_TIMEOUT}s, fallback ke polling")
        return None
    except Exception as e:
        print(f"[{label}] runsync error: {e}, fallback ke polling")
        return None


async def _run_poll(image, mask, label="", http_session=None) -> Optional[Image.Image]:
    img_send, mask_send, sw, sh = _prepare_slice(image, mask)
    payload = _build_payload(img_send, mask_send)
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type":  "application/json",
    }
    start = time.time()

    try:
        async with http_session.post(
            f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run",
            headers=headers, json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as r:
            if r.status != 200:
                print(f"[{label}] poll submit HTTP {r.status}")
                return None
            data = await r.json()
    except Exception as e:
        print(f"[{label}] poll submit error: {e}")
        return None

    job_id = data.get("id")
    if not job_id:
        return None

    for _ in range(POLL_MAX):
        await asyncio.sleep(POLL_INTERVAL)
        try:
            async with http_session.get(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as s:
                if s.status != 200:
                    continue
                sj = await s.json()
        except Exception:
            continue

        status = sj.get("status")
        if status == "COMPLETED":
            img     = await decode_image_async(sj.get("output"), http_session=http_session)
            elapsed = time.time() - start
            # FIXED BUG2: guard img is not None sebelum .size
            if img is None:
                print(f"[{label}] poll: output tidak bisa di-decode ({elapsed:.2f}s)")
                return None
            if img.size != (sw, sh):
                img = img.crop((0, 0, sw, sh))
            print(f"[{label}] poll selesai dalam {elapsed:.2f}s")
            return img
        elif status in ("FAILED", "CANCELLED"):
            print(f"[{label}] poll {status}: {sj.get('error')} ({time.time()-start:.2f}s)")
            return None

    print(f"[{label}] poll timeout ({POLL_MAX * POLL_INTERVAL}s)")
    return None


async def run_runpod_lama(
    image,
    mask,
    label="",
    http_session=None,
) -> Optional[Image.Image]:
    if http_session is None:
        async with aiohttp.ClientSession() as session:
            return await run_runpod_lama(image, mask, label, session)

    orig_w, orig_h = image.size
    cols = math.ceil(orig_w / TILE_SIZE)
    rows = math.ceil(orig_h / TILE_SIZE)
    
    full_result = image.copy()
    
    print(f"[{label}] Memulai grid slicing {rows}x{cols} tiles ({TILE_SIZE}px).")

    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * TILE_SIZE, r * TILE_SIZE
            x2, y2 = min(x1 + TILE_SIZE, orig_w), min(y1 + TILE_SIZE, orig_h)
            
            slice_img = image.crop((x1, y1, x2, y2))
            slice_mask = mask.crop((x1, y1, x2, y2))
            
            # Optimasi: Lewati jika tidak ada masker di tile ini
            if slice_mask.getextrema() == (0, 0):
                continue
                
            print(f"  > Memproses tile ({r},{c}) {slice_img.size[0]}x{slice_img.size[1]}.")
            
            async with (runpod_sem or asyncio.Lock()):
                # Gunakan runsync untuk tile kecil agar cepat, fallback ke poll jika gagal
                inpainted_tile = await _run_runsync(
                    slice_img, slice_mask, 
                    label=f"{label}_t{r}{c}", 
                    http_session=http_session
                )
                
                if inpainted_tile is None:
                    inpainted_tile = await _run_poll(
                        slice_img, slice_mask, 
                        label=f"{label}_t{r}{c}", 
                        http_session=http_session
                    )
            
            if inpainted_tile:
                full_result.paste(inpainted_tile, (x1, y1))
                
    print(f"[{label}] Selesai pemrosesan grid slicing.")
    return full_result