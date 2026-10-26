# ============================================================
# core/pipeline.py
# RunPod handles YOLO + LaMa sekaligus
# Railway hanya: download → kirim ke RunPod → upload hasil
# ============================================================

import gc
import io
import json
import time
import asyncio
import traceback
from datetime import datetime, timedelta
from typing import Optional
import os

import aiohttp
from PIL import Image
from googleapiclient.errors import HttpError

from core.config import (
    MAX_WIDTH,
    DRIVE_OUTPUT_FOLDER_ID,
    OUTPUT_DELETE_DELAY_MINUTES,
)
from core.database import (
    db_get_job, db_update_job, db_append_log, db_mark_file_processed,
    db_get_processed_files, db_schedule_deletion, db_reset_processed_files,
    db_record_file_failure, db_clear_file_failure, db_remove_pending_deletion,
)
from core.drive import (
    _get_drive, filter_and_sort_files, create_output_folder,
    upload_file, download_file_async, extract_folder_id, delete_folder,
)
from core.image_processing import to_bytes, progress_bar, format_eta, get_dynamic_batch_size
from core.runpod_client import run_runpod_lama

# ── Shared State ──────────────────────────────────────────
pipeline_sem:  Optional[asyncio.Semaphore]     = None
_warmup_lock:  Optional[asyncio.Lock]          = None
_http_session: Optional[aiohttp.ClientSession] = None
_runpod_warmed = False

jobs           = {}
job_queue      = []
cancelled_jobs = set()

LOG_MAX_ENTRIES = 100
FETCH_TIMEOUT   = 90


def job_log(job_id, message):
    print(message)
    db_append_log(job_id, message)
    if job_id in jobs:
        jobs[job_id]["progress"] = message
        log = jobs[job_id]["log"]
        log.append(message)
        if len(log) > LOG_MAX_ENTRIES:
            jobs[job_id]["log"] = log[-LOG_MAX_ENTRIES:]


def get_queue_position(job_id):
    try:
        return job_queue.index(job_id) + 1
    except ValueError:
        return 0


# ── Warmup ────────────────────────────────────────────────

async def warmup(job_id=None):
    global _runpod_warmed
    async with _warmup_lock:
        if _runpod_warmed:
            return
        msg = "🔥 Warming up RunPod..."
        print(msg)
        if job_id:
            job_log(job_id, msg)
        try:
            from core.config import RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID
            import base64 as _b64

            tiny = Image.new("RGB", (128, 128), color=(255, 255, 255))
            buf  = io.BytesIO()
            tiny.save(buf, format="JPEG")
            payload = {
                "input": {
                    "image": "data:image/jpeg;base64," + _b64.b64encode(buf.getvalue()).decode(),
                }
            }
            headers = {
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
                "Content-Type":  "application/json",
            }
            t0 = time.time()
            try:
                async with _http_session.post(
                    f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync",
                    headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as r:
                    elapsed  = time.time() - t0
                    status   = (await r.json()).get("status", "?") if r.status == 200 else f"HTTP {r.status}"
                    done_msg = f"✅ RunPod warm ({elapsed:.1f}s) — {status}"
            except asyncio.TimeoutError:
                done_msg = "⚠️ Warmup timeout — lanjut saja"
            except Exception as e:
                done_msg = f"⚠️ Warmup error ({e})"
            print(done_msg)
            if job_id:
                job_log(job_id, done_msg)
        except Exception as e:
            print(f"⚠️ Warmup skip: {e}")
        finally:
            _runpod_warmed = True


# ── Process Single Image ──────────────────────────────────

async def process_image(
    file_id:          str,
    filename:         str,
    output_folder_id: str,
    job_id:           str,
    local_output_dir: str = None,
) -> str:
    start_time = time.time()

    try:
        if job_id in cancelled_jobs:
            return "cancelled"

        job_log(job_id, f"Processing {filename}")

        # Download gambar
        raw = await asyncio.wait_for(
            download_file_async(file_id, _http_session),
            timeout=FETCH_TIMEOUT,
        )

        # Resize jika terlalu lebar
        def load_image():
            img = Image.open(raw).convert("RGB")
            if img.width > MAX_WIDTH:
                ratio = MAX_WIDTH / img.width
                img   = img.resize(
                    (MAX_WIDTH, int(img.height * ratio)), Image.LANCZOS
                )
            return img

        img = await asyncio.to_thread(load_image)

        # Kirim ke RunPod — YOLO + LaMa sekalian di sana
        result = await run_runpod_lama(
            img, None,
            label=filename,
            http_session=_http_session,
        )

        if result is None:
            raise Exception("RunPod returned None")

        # Keep the source file extension and actual encoded format in sync.
        ext = os.path.splitext(filename)[1].lower()
        format_by_ext = {
            ".jpg": ("JPEG", "image/jpeg"),
            ".jpeg": ("JPEG", "image/jpeg"),
            ".png": ("PNG", "image/png"),
            ".webp": ("WEBP", "image/webp"),
        }
        image_format, mimetype = format_by_ext.get(ext, ("JPEG", "image/jpeg"))
        out_buf = to_bytes(result, fmt=image_format, quality=95)
        if local_output_dir:
            out_path = os.path.join(local_output_dir, os.path.basename(filename))
            with open(out_path, "wb") as f:
                f.write(out_buf.getvalue())
        else:
            await asyncio.to_thread(
                upload_file, out_buf, filename, output_folder_id, mimetype
            )

        duration = time.time() - start_time
        completed = db_mark_file_processed(job_id, filename, "success", duration)
        db_clear_file_failure(job_id, filename)
        if job_id in jobs:
            jobs[job_id]["completed_files"] = completed

        row = db_get_job(job_id)
        job_log(job_id,
            f"{progress_bar(completed, row['total_files'] or 0)} "
            f"{filename} ({duration:.1f}s)"
        )
        return "success"

    except Exception as e:
        duration = time.time() - start_time
        completed = db_mark_file_processed(job_id, filename, "failed", duration)
        db_record_file_failure(job_id, filename)
        if job_id in jobs:
            jobs[job_id]["completed_files"] = completed

        job_log(job_id, f"Error on {filename}: {e}")
        print(traceback.format_exc())
        return "failed"

    finally:
        gc.collect()


# ── Pipeline ──────────────────────────────────────────────

async def pipeline(job_id: str, folder_url: str):
    output_folder_id = None

    try:
        db_update_job(job_id, status="running")
        if job_id in jobs:
            jobs[job_id]["status"] = "running"

        folder_id = extract_folder_id(folder_url)
        if not folder_id or len(folder_id) < 5:
            raise Exception(f"ID Folder tidak valid: '{folder_id}'")

        job_log(job_id, f"Reading folder: {folder_id}")
        await warmup(job_id)

        def _list_folder():
            return _get_drive().files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="files(id, name)"
            ).execute()

        result               = await asyncio.to_thread(_list_folder)
        all_files            = result.get("files", [])
        valid_files, skipped = filter_and_sort_files(all_files)

        if skipped:
            job_log(job_id, f"Skipped: {skipped}")
        if not valid_files:
            raise Exception("No valid images found")

        # Reuse the previous output folder for resumable jobs. If it no longer
        # exists (for example, its 24-hour retention expired), restart from scratch.
        job_row = db_get_job(job_id)
        previous_output_id = job_row["output_folder_id"] if job_row else None
        folder_name = None

        if previous_output_id:
            try:
                output_meta = await asyncio.to_thread(
                    lambda: _get_drive().files().get(
                        fileId=previous_output_id,
                        fields="id,name,mimeType,trashed",
                    ).execute()
                )
            except HttpError as e:
                if getattr(e, "resp", None) is not None and e.resp.status == 404:
                    output_meta = None
                else:
                    raise

            if (
                output_meta
                and not output_meta.get("trashed", False)
                and output_meta.get("mimeType") == "application/vnd.google-apps.folder"
            ):
                output_folder_id = previous_output_id
                folder_name = job_row["result_folder"] or output_meta.get("name")
                db_remove_pending_deletion(output_folder_id)
            else:
                db_remove_pending_deletion(previous_output_id)
                db_reset_processed_files(job_id)

        if not output_folder_id:
            # No reusable output folder means prior success records cannot be trusted.
            db_reset_processed_files(job_id)
            folder_name, output_folder_id = await asyncio.to_thread(
                create_output_folder, DRIVE_OUTPUT_FOLDER_ID, "output"
            )

        db_update_job(
            job_id,
            total_files=len(valid_files),
            output_folder_id=output_folder_id,
            result_folder=folder_name,
        )
        if job_id in jobs:
            jobs[job_id]["total_files"]      = len(valid_files)
            jobs[job_id]["output_folder_id"] = output_folder_id
            jobs[job_id]["result_folder"]    = folder_name

        available_names = {f["name"] for f in valid_files}
        already_done = db_get_processed_files(job_id) & available_names
        pending_files = [f for f in valid_files if f["name"] not in already_done]

        if already_done:
            job_log(job_id, f"Resume: {len(already_done)} done, {len(pending_files)} remaining")

        batch_size    = max(1, get_dynamic_batch_size(len(pending_files)))
        success_count = len(already_done)
        failed_count  = 0
        file_times    = []

        job_log(job_id, f"Total: {len(pending_files)} files | Batch: {batch_size}")

        for i in range(0, len(pending_files), batch_size):
            if job_id in cancelled_jobs:
                job_log(job_id, "Cancelled by user")
                db_update_job(job_id, status="cancelled")
                if job_id in jobs:
                    jobs[job_id]["status"] = "cancelled"
                break

            batch = pending_files[i:i + batch_size]
            t0    = time.time()

            results = await asyncio.gather(*[
                process_image(f["id"], f["name"], output_folder_id, job_id)
                for f in batch
            ])

            batch_time = time.time() - t0
            file_times.append(batch_time / max(len(batch), 1))

            for r in results:
                if r == "success":  success_count += 1
                elif r == "failed": failed_count  += 1

            if file_times:
                avg         = sum(file_times) / len(file_times)
                remaining   = len(pending_files) - (i + len(batch))
                eta_seconds = int(avg * remaining)
                if job_id in jobs:
                    jobs[job_id]["eta"] = format_eta(eta_seconds)

        # Finalize
        current_status = db_get_job(job_id)["status"]
        if current_status not in ("cancelled",):
            all_failed = success_count == 0 and (
                failed_count > 0 or bool(pending_files)
            )
            if all_failed:
                final_status = "failed"
            elif failed_count > 0:
                final_status = "completed_with_errors"
            else:
                final_status = "completed"
            db_update_job(job_id, status=final_status, finished_at=datetime.now().isoformat())
            if job_id in jobs:
                jobs[job_id]["status"] = final_status

        if output_folder_id:
            db_schedule_deletion(output_folder_id, OUTPUT_DELETE_DELAY_MINUTES)
            delete_at = datetime.now().astimezone() + timedelta(
                minutes=OUTPUT_DELETE_DELAY_MINUTES
            )
            if job_id in jobs:
                jobs[job_id]["delete_at_unix"] = int(delete_at.timestamp())

        job_log(job_id, f"Done! ✅ {success_count} berhasil | ❌ {failed_count} gagal")

    except Exception as e:
        db_update_job(job_id, status="failed", finished_at=datetime.now().isoformat())
        if job_id in jobs:
            jobs[job_id]["status"] = "failed"
        # Preserve partial output for the same retention period after a failed job.
        if output_folder_id:
            db_schedule_deletion(output_folder_id, OUTPUT_DELETE_DELAY_MINUTES)
            delete_at = datetime.now().astimezone() + timedelta(
                minutes=OUTPUT_DELETE_DELAY_MINUTES
            )
            if job_id in jobs:
                jobs[job_id]["delete_at_unix"] = int(delete_at.timestamp())
        job_log(job_id, f"Pipeline failed: {e}")
        print(traceback.format_exc())

    finally:
        if job_id in job_queue:
            job_queue.remove(job_id)
        cancelled_jobs.discard(job_id)
        gc.collect()


async def run_pipeline(job_id: str, folder_url: str):
    job_queue.append(job_id)
    pos = get_queue_position(job_id)
    if pos > 1:
        db_update_job(job_id, status="queued")
        if job_id in jobs:
            jobs[job_id]["status"]         = "queued"
            jobs[job_id]["queue_position"] = pos
        job_log(job_id, f"Queued at position #{pos}")

    async with pipeline_sem:
        db_update_job(job_id, status="running")
        if job_id in jobs:
            jobs[job_id]["status"]         = "running"
            jobs[job_id]["queue_position"] = 0
        await pipeline(job_id, folder_url)

    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


async def deletion_loop():
    from core.database import db_get_pending_deletions, db_remove_pending_deletion
    while True:
        await asyncio.sleep(60)
        try:
            for folder_id in db_get_pending_deletions():
                success = await asyncio.to_thread(delete_folder, folder_id)
                if success:
                    db_remove_pending_deletion(folder_id)
                    print(f"🗑️ Auto-deleted: {folder_id}")
                else:
                    print(f"Delete failed, will retry: {folder_id}")
        except Exception as e:
            print(f"Deletion loop error: {e}")
