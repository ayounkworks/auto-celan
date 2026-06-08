# ============================================================
# core/runpod_client.py
# FIXED v2:
# - BUG2: Guard img is not None sebelum img.size (line 160, 234)
# ============================================================

import asyncio
import base64
import time
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
LAMA_MAX_AREA = 800 * 20000


def _resize_for_lama(image, mask):
    orig_w, orig_h = image.size
    tw = max(LAMA_MIN_SIZE, (orig_w // 8) * 8)
    th = max(LAMA_MIN_SIZE, (orig_h // 8) * 8)

    if (tw, th) == (orig_w, orig_h):
        return image, mask, orig_w, orig_h, False
    return (
        image.resize((tw, th), Image.Resampling.LANCZOS),
        mask.resize((tw, th), Image.Resampling.NEAREST),
        orig_w, orig_h, True,
    )


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
    img_send, mask_send, orig_w, orig_h, was_resized = _resize_for_lama(image, mask)
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
            if was_resized and img.size != (orig_w, orig_h):
                img = img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
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
    img_send, mask_send, orig_w, orig_h, was_resized = _resize_for_lama(image, mask)
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
            if was_resized and img.size != (orig_w, orig_h):
                img = img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
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

    area = image.width * image.height
    async with (runpod_sem or asyncio.Lock()):
        if area >= 800 * 4000:
            print(f"  [{label}] gambar besar ({image.width}x{image.height}), pakai polling")
            return await _run_poll(image, mask, label, http_session)
        result = await _run_runsync(image, mask, label, http_session)
        if result is not None:
            return result
        return await _run_poll(image, mask, label, http_session)