# ============================================================
# core/runpod_client.py
# Railway hanya kirim image → RunPod detect + inpaint → return hasil
# Tidak ada mask dari Railway — mask dibuat di RunPod oleh YOLO
# ============================================================

import asyncio
import base64
import time
from io import BytesIO
from typing import Optional

import aiohttp
from PIL import Image

from core.config import RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID
from core.retry_policy import is_retryable_http_status, poll_backoff_seconds

runpod_sem: Optional[asyncio.Semaphore] = None

POLL_INTERVAL   = 1.5
POLL_MAX        = 180
MAX_JOB_RETRIES = 1
RETRY_BACKOFF   = 2.0
MAX_CONSECUTIVE_POLL_ERRORS = 5


def normalize_b64(data: str):
    data = data.strip()
    if "," in data:
        data = data.split(",", 1)[1].strip()
    data += "=" * (-len(data) % 4)
    return data


def _build_payload(image: Image.Image, job_id=None, filename=None) -> dict:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return {
        "input": {
            "image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
            "job_id": job_id,
            "filename": filename,
        }
    }


async def _decode_result(output, http_session) -> Image.Image:
    if not output:
        raise RuntimeError("RunPod job completed without an output payload")
    if isinstance(output, dict):
        if output.get("error"):
            raise RuntimeError(f"RunPod handler error: {output['error']}")
        img_data = output.get("image")
    elif isinstance(output, str):
        img_data = output
    else:
        raise RuntimeError(f"Unsupported RunPod output type: {type(output).__name__}")
    if not img_data:
        raise RuntimeError("RunPod output does not contain an image")

    if isinstance(img_data, str) and img_data.startswith("http"):
        async with http_session.get(img_data, timeout=aiohttp.ClientTimeout(total=60)) as r:
            if r.status != 200:
                body = (await r.text())[:500]
                raise RuntimeError(f"RunPod result URL HTTP {r.status}: {body}")
            raw = await r.read()
    else:
        try:
            raw = base64.b64decode(normalize_b64(img_data), validate=True)
        except Exception as e:
            raise RuntimeError(f"Invalid base64 image in RunPod output: {e}") from e
    try:
        image = Image.open(BytesIO(raw))
        image.load()
        return image.convert("RGB")
    except Exception as e:
        raise RuntimeError(f"RunPod returned unreadable image ({len(raw)} bytes): {e}") from e


def _is_retryable_worker_error(message: str) -> bool:
    message = (message or "").lower()
    markers = ("worker startup", "worker unavailable", "temporarily unavailable",
               "gpu unavailable", "connection reset", "internal server error",
               "worker timeout", "timed out")
    return any(marker in message for marker in markers)


async def _run_poll(
    image: Image.Image, label="", http_session=None, job_id=None, filename=None
) -> Image.Image:
    payload = _build_payload(image, job_id=job_id, filename=filename)
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    start = time.time()

    for attempt in range(MAX_JOB_RETRIES + 1):
        try:
            async with http_session.post(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run",
                headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60),
            ) as r:
                if r.status != 200:
                    body = (await r.text())[:1000]
                    if is_retryable_http_status(r.status) and attempt < MAX_JOB_RETRIES:
                        await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
                        continue
                    raise RuntimeError(f"RunPod submit HTTP {r.status}: {body}")
                data = await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Submission may have succeeded despite a client-side timeout. Do not duplicate it.
            raise RuntimeError(f"RunPod submit outcome unknown; not resubmitting: {e}") from e

        job_id = data.get("id")
        if not job_id:
            raise RuntimeError(f"RunPod response has no job ID: {str(data)[:1000]}")

        terminal_error = None
        consecutive_poll_errors = 0
        for poll_index in range(POLL_MAX):
            await asyncio.sleep(POLL_INTERVAL)
            try:
                async with http_session.get(
                    f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}",
                    headers=headers, timeout=aiohttp.ClientTimeout(total=20),
                ) as s:
                    if s.status != 200:
                        body = (await s.text())[:500]
                        if not is_retryable_http_status(s.status):
                            raise RuntimeError(
                                f"RunPod status HTTP {s.status} job={job_id}: {body}"
                            )
                        consecutive_poll_errors += 1
                        if consecutive_poll_errors > MAX_CONSECUTIVE_POLL_ERRORS:
                            raise RuntimeError(
                                f"RunPod status polling failed {consecutive_poll_errors} "
                                f"consecutive times for job={job_id}; job may still be running. "
                                "Check RunPod status before retrying."
                            )
                        delay = poll_backoff_seconds(consecutive_poll_errors)
                        print(
                            f"[{label}] transient status HTTP {s.status} "
                            f"job={job_id}; retry {consecutive_poll_errors}/"
                            f"{MAX_CONSECUTIVE_POLL_ERRORS} in {delay:.1f}s: {body}"
                        )
                        await asyncio.sleep(delay)
                        continue
                    sj = await s.json()
                    consecutive_poll_errors = 0
            except RuntimeError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                consecutive_poll_errors += 1
                if consecutive_poll_errors > MAX_CONSECUTIVE_POLL_ERRORS:
                    raise RuntimeError(
                        f"RunPod status polling had {consecutive_poll_errors} "
                        f"consecutive network errors for job={job_id}; job may still "
                        "be running. Check RunPod status before retrying."
                    ) from e
                delay = poll_backoff_seconds(consecutive_poll_errors)
                print(
                    f"[{label}] status poll network error job={job_id}; retry "
                    f"{consecutive_poll_errors}/{MAX_CONSECUTIVE_POLL_ERRORS} "
                    f"in {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)
                continue

            status = sj.get("status")
            if status == "COMPLETED":
                result = await _decode_result(sj.get("output"), http_session)
                print(f"[{label}] poll done {time.time()-start:.2f}s job={job_id} polls={poll_index+1}")
                return result
            if status in ("FAILED", "CANCELLED"):
                terminal_error = str(sj.get("error") or sj.get("output") or status)
                print(f"[{label}] poll {status} job={job_id}: {terminal_error[:1000]}")
                break

        if terminal_error is None:
            raise TimeoutError(
                f"RunPod job {job_id} did not reach a terminal state within "
                f"{POLL_MAX * POLL_INTERVAL:.0f}s; check its status before retrying"
            )
        if attempt < MAX_JOB_RETRIES and _is_retryable_worker_error(terminal_error):
            await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
            continue
        raise RuntimeError(f"RunPod job {job_id} ended unsuccessfully: {terminal_error}")

    raise RuntimeError("RunPod retry budget exhausted")


async def run_runpod_lama(
    image: Image.Image,
    mask,           # diabaikan — mask dibuat di RunPod oleh YOLO
    label="",
    http_session=None,
    job_id=None,
    filename=None,
) -> Optional[Image.Image]:
    if http_session is None:
        async with aiohttp.ClientSession() as session:
            return await run_runpod_lama(image, mask, label, session, job_id, filename)

    # The application injects a shared semaphore at startup. A fresh lock per
    # call would not limit concurrency across simultaneous requests.
    if runpod_sem is None:
        return await _run_poll(image, label, http_session, job_id, filename)
    async with runpod_sem:
        return await _run_poll(image, label, http_session, job_id, filename)