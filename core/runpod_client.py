# ============================================================
# core/runpod_client.py
# Railway hanya kirim image → RunPod detect + inpaint → return hasil
# Tidak ada mask dari Railway — mask dibuat di RunPod oleh YOLO
# ============================================================

import asyncio
import base64
import time
import math
from io import BytesIO
from typing import Optional

import aiohttp
from PIL import Image

from core.config import RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID

runpod_sem: Optional[asyncio.Semaphore] = None

RUNSYNC_TIMEOUT = 180
POLL_INTERVAL   = 1.0
POLL_MAX        = 180


def normalize_b64(data: str):
    data = data.strip()
    if "," in data:
        data = data.split(",", 1)[1].strip()
    data += "=" * (-len(data) % 4)
    return data


def _build_payload(image: Image.Image) -> dict:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return {
        "input": {
            "image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
        }
    }


async def _decode_result(output, http_session) -> Optional[Image.Image]:
    if not output:
        return None

    # Extract image string dari output
    if isinstance(output, dict):
        img_data = output.get("image")
    elif isinstance(output, str):
        img_data = output
    else:
        return None

    if not img_data:
        return None

    # URL
    if img_data.startswith("http"):
        async with http_session.get(img_data, timeout=aiohttp.ClientTimeout(total=60)) as r:
            if r.status != 200:
                return None
            return Image.open(BytesIO(await r.read())).convert("RGB")

    # Base64
    try:
        b64 = normalize_b64(img_data)
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        print(f"[runpod_client] decode error: {e}")
        return None


async def _run_runsync(image: Image.Image, label="", http_session=None) -> Optional[Image.Image]:
    payload = _build_payload(image)
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
            img = await _decode_result(data.get("output"), http_session)
            print(f"[{label}] runsync done {time.time()-start:.2f}s boxes={data.get('output', {}).get('boxes', '?')}")
            return img
        elif status == "FAILED":
            print(f"[{label}] runsync FAILED: {data.get('error')}")
            return None
        else:
            print(f"[{label}] runsync status={status}, fallback ke polling")
            return None

    except asyncio.TimeoutError:
        print(f"[{label}] runsync timeout {RUNSYNC_TIMEOUT}s, fallback polling")
        return None
    except Exception as e:
        print(f"[{label}] runsync error: {e}, fallback polling")
        return None


async def _run_poll(image: Image.Image, label="", http_session=None) -> Optional[Image.Image]:
    payload = _build_payload(image)
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
            img = await _decode_result(sj.get("output"), http_session)
            print(f"[{label}] poll done {time.time()-start:.2f}s")
            return img
        elif status in ("FAILED", "CANCELLED"):
            print(f"[{label}] poll {status}: {sj.get('error')}")
            return None

    print(f"[{label}] poll timeout")
    return None


async def run_runpod_lama(
    image: Image.Image,
    mask,           # diabaikan — mask dibuat di RunPod oleh YOLO
    label="",
    http_session=None,
) -> Optional[Image.Image]:
    if http_session is None:
        async with aiohttp.ClientSession() as session:
            return await run_runpod_lama(image, mask, label, session)

    async with (runpod_sem or asyncio.Lock()):
        for attempt in range(3):
            result = await _run_runsync(image, label, http_session)
            if result is not None:
                return result
            result = await _run_poll(image, label, http_session)
            if result is not None:
                return result
            if attempt < 2:
                print(f"[{label}] attempt {attempt + 1} failed, retrying in 3s...")
                await asyncio.sleep(3)
        return None