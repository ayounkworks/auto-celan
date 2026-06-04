# ============================================================
# core/pipeline.py
# Orchestrasi: download → Vision → smart_clean → RunPod → upload
#
# FIXES:
# 1. completed_files increment atomic (pakai db UPDATE SET x=x+1)
# 2. fetch_vision() punya per-file timeout → satu file hang tidak blokir batch
# 3. process_image tidak double-download jika img_and_texts sudah ada
# 4. solid_fill_inpaint dipanggil dengan koordinat yang benar (0,0 karena full image)
# 5. Log lebih informatif + ETA lebih akurat
# ============================================================

import gc
import io
import json
import time
import asyncio
import traceback
from datetime import datetime
from typing import Optional

import aiohttp
import numpy as np
from PIL import Image, ImageFilter
from google.cloud import vision

from core.config import MAX_WIDTH, DRIVE_OUTPUT_FOLDER_ID
from core.database import (
    db_get_job, db_update_job, db_append_log, db_mark_file_processed,
    db_get_processed_files, db_schedule_deletion, db_increment_completed,
)
from core.drive import (
    _get_drive, CREDS, filter_and_sort_files, create_output_folder,
    upload_file, download_file_async, extract_folder_id,
)
from core.image_processing import (
    to_bytes, progress_bar, format_eta, get_dynamic_batch_size,
    smart_clean, validate_inpaint, solid_fill_inpaint,
)
from core.runpod_client import run_runpod_lama

# ── Shared State (diisi oleh lifespan di main.py) ─────────
pipeline_sem:  Optional[asyncio.Semaphore] = None
vision_sem:    Optional[asyncio.Semaphore] = None
_warmup_lock:  Optional[asyncio.Lock]      = None
_http_session: Optional[aiohttp.ClientSession] = None
_runpod_warmed = False

# In-memory job state
jobs           = {}
job_queue      = []
cancelled_jobs = set()

# Vision client (diinit di main.py setelah config siap)
vision_client = None

# Timeout per file untuk fetch+vision (detik)
FETCH_VISION_TIMEOUT = 90


def job_log(job_id, message):
    print(message)
    db_append_log(job_id, message)
    if job_id in jobs:
        jobs[job_id]["progress"] = message
        jobs[job_id]["log"].append(message)


def get_queue_position(job_id):
    try:
        return job_queue.index(job_id) + 1
    except ValueError:
        return 0


# ── Warmup ────────────────────────────────────────────────

async def warmup(job_id=None):
    """
    Kirim request dummy 1x1 pixel ke RunPod agar worker 'bangun'
    sebelum file pertama diproses. Cold start (~3–10 detik) terjadi
    di sini, bukan di file pertama yang penting.
    """
    global _runpod_warmed
    async with _warmup_lock:
        if _runpod_warmed:
            return

        msg = "🔥 Warming up RunPod worker..."
        print(msg)
        if job_id:
            job_log(job_id, msg)

        try:
            from PIL import Image as _Image
            import io as _io
            import base64 as _b64
            from core.config import RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID

            # Buat gambar 128x128 pixel putih + mask hitam
            tiny_img  = _Image.new("RGB", (128, 128), color=(255, 255, 255))
            tiny_mask = _Image.new("L",   (128, 128), color=0)

            def _to_b64(img, fmt):
                buf = _io.BytesIO()
                img.save(buf, format=fmt)
                return _b64.b64encode(buf.getvalue()).decode()

            payload = {
                "input": {
                    "image": "data:image/jpeg;base64," + _to_b64(tiny_img,  "JPEG"),
                    "mask":  "data:image/png;base64,"  + _to_b64(tiny_mask, "PNG"),
                }
            }
            headers = {
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
                "Content-Type":  "application/json",
            }

            # Pakai /runsync dengan timeout longgar — kita tidak peduli hasilnya,
            # hanya ingin worker aktif sebelum batch dimulai
            t0 = __import__("time").time()
            try:
                async with _http_session.post(
                    f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as r:
                    elapsed = __import__("time").time() - t0
                    status  = (await r.json()).get("status", "?") if r.status == 200 else f"HTTP {r.status}"
                    done_msg = f"✅ RunPod warm ({elapsed:.1f}s) — status: {status}"
            except asyncio.TimeoutError:
                done_msg = "⚠️ Warmup timeout — worker mungkin masih cold, lanjut saja"
            except Exception as e:
                done_msg = f"⚠️ Warmup error ({e}) — lanjut saja"

            print(done_msg)
            if job_id:
                job_log(job_id, done_msg)

        except Exception as e:
            # Warmup gagal tidak boleh hentikan pipeline
            warn = f"⚠️ Warmup skip: {e}"
            print(warn)
            if job_id:
                job_log(job_id, warn)

        finally:
            _runpod_warmed = True


# ── Process Single Image ──────────────────────────────────

async def process_image(
    file_id:          str,
    filename:         str,
    output_folder_id: str,
    job_id:           str,
    img_and_texts:    Optional[tuple] = None,
    local_output_dir: str = None,
) -> str:
    start_time = time.time()
    img        = None
    final      = None
    inpaint    = None

    try:
        if job_id in cancelled_jobs:
            return "cancelled"

        # img_and_texts sudah diprefetch oleh fetch_vision() di pipeline
        if img_and_texts is None:
            # Fallback: fetch sendiri jika dipanggil langsung
            job_log(job_id, f"Processing {filename}")
            raw = await download_file_async(file_id, _http_session)

            async with vision_sem:
                def resize_and_detect():
                    image = Image.open(raw).convert("RGB")
                    raw.close()
                    if image.width > MAX_WIDTH:
                        ratio = MAX_WIDTH / image.width
                        image = image.resize(
                            (MAX_WIDTH, int(image.height * ratio)), Image.LANCZOS
                        )
                    buf = to_bytes(image, "JPEG", 95)
                    buf.seek(0)
                    response = vision_client.document_text_detection(
                        image=vision.Image(content=buf.read())
                    )
                    return image, response.text_annotations

                img, texts = await asyncio.to_thread(resize_and_detect)
        else:
            img, texts = img_and_texts

        if not texts:
            out_buf = to_bytes(img)
            await _save_output(out_buf, filename, output_folder_id, local_output_dir)
            status = "skip"

        else:
            img_np = np.array(img)

            prefilled, lama_mask, sfx_count, dialog_count = await asyncio.to_thread(
                smart_clean, img, texts, img_np
            )
            del img_np

            has_mask = lama_mask.getbbox() is not None

            if has_mask:
                # Coba solid fill dulu (tanpa RunPod) — cl=0, ct=0 karena ini full image
                solid_result = await asyncio.to_thread(
                    solid_fill_inpaint, prefilled, lama_mask, prefilled, 0, 0
                )

                if solid_result is not None:
                    job_log(job_id, f"  {filename}: solid fill (no RunPod)")
                    final = solid_result
                else:
                    raw_inpaint = await run_runpod_lama(
                        prefilled, lama_mask,
                        label=filename,
                        http_session=_http_session,
                    )
                    inpaint = validate_inpaint(raw_inpaint, prefilled)
                    if inpaint is not None:
                        soft_mask = lama_mask.filter(ImageFilter.GaussianBlur(7))
                        final     = prefilled.copy()
                        final.paste(inpaint, (0, 0), soft_mask)
                    else:
                        raise Exception("Inpaint result corrupt, file skipped")
            else:
                final = prefilled

            if sfx_count or dialog_count:
                job_log(job_id,
                    f"  {filename}: {dialog_count} dialog, {sfx_count} SFX skipped"
                )

            out_buf = to_bytes(final)
            await _save_output(out_buf, filename, output_folder_id, local_output_dir)
            status = "success"

        duration = time.time() - start_time
        db_mark_file_processed(job_id, filename, status, duration)

        # FIX: atomic increment agar tidak race condition di concurrent tasks
        completed = db_increment_completed(job_id)
        if job_id in jobs:
            jobs[job_id]["completed_files"] = completed

        row = db_get_job(job_id)
        job_log(job_id,
            f"{progress_bar(completed, row['total_files'] or 0)} "
            f"{filename} ({duration:.1f}s)"
        )
        return status

    except Exception as e:
        duration  = time.time() - start_time
        completed = db_increment_completed(job_id)
        if job_id in jobs:
            jobs[job_id]["completed_files"] = completed

        row      = db_get_job(job_id)
        failed   = json.loads(row["failed_files"] or "[]")
        failed.append(filename)
        db_update_job(job_id, failed_files=json.dumps(failed))
        db_mark_file_processed(job_id, filename, "failed", duration)

        job_log(job_id, f"Error on {filename}: {e}")
        print(traceback.format_exc())
        return "failed"

    finally:
        for obj in [img, inpaint, final]:
            try:
                if obj:
                    obj.close()
            except Exception:
                pass
        gc.collect()


async def _save_output(out_buf, filename, output_folder_id, local_output_dir):
    """Helper: simpan ke lokal atau upload ke Drive."""
    if local_output_dir:
        import os
        out_path = os.path.join(local_output_dir, filename)
        with open(out_path, "wb") as f:
            f.write(out_buf.getvalue())
    else:
        await asyncio.to_thread(
            upload_file, out_buf, filename, output_folder_id, "image/jpeg"
        )


# ── Pipeline ──────────────────────────────────────────────

async def pipeline(job_id: str, folder_url: str):
    output_folder_id = None

    try:
        db_update_job(job_id, status="running")
        if job_id in jobs:
            jobs[job_id]["status"] = "running"

        folder_id = extract_folder_id(folder_url)
        if not folder_id or len(folder_id) < 5:
            raise Exception(f"ID Folder Google Drive tidak valid: '{folder_id}'")

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
            job_log(job_id, f"Skipped non-image: {skipped}")
        if not valid_files:
            raise Exception("No valid images found in folder")

        import os

        def _get_folder_name():
            meta = _get_drive().files().get(fileId=folder_id, fields="name").execute()
            return meta.get("name", folder_id)

        input_folder_name = await asyncio.to_thread(_get_folder_name)
        folder_name, output_folder_id = await asyncio.to_thread(
            create_output_folder, DRIVE_OUTPUT_FOLDER_ID, input_folder_name
        )
        local_output_dir = None  # None = upload ke Drive

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

        already_done  = db_get_processed_files(job_id)
        pending_files = [f for f in valid_files if f["name"] not in already_done]

        if already_done:
            job_log(job_id,
                f"Resume: {len(already_done)} done, {len(pending_files)} remaining"
            )

        file_times    = []
        success_count = len(already_done)
        failed_count  = 0
        skip_count    = 0

        batch_size = get_dynamic_batch_size(len(pending_files))
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

            # ── Phase 1: Download + Vision paralel dengan timeout per-file ──
            async def fetch_vision(f):
                try:
                    job_log(job_id, f"Fetching {f['name']}")
                    raw = await asyncio.wait_for(
                        download_file_async(f["id"], _http_session),
                        timeout=FETCH_VISION_TIMEOUT,
                    )
                    async with vision_sem:
                        def resize_and_detect():
                            image = Image.open(raw).convert("RGB")
                            if image.width > MAX_WIDTH:
                                ratio = MAX_WIDTH / image.width
                                image = image.resize(
                                    (MAX_WIDTH, int(image.height * ratio)), Image.LANCZOS
                                )
                            buf = to_bytes(image, "JPEG", 95)
                            buf.seek(0)
                            response = vision_client.document_text_detection(
                                image=vision.Image(content=buf.read())
                            )
                            raw.close()
                            return image, response.text_annotations

                        return await asyncio.wait_for(
                            asyncio.to_thread(resize_and_detect),
                            timeout=FETCH_VISION_TIMEOUT,
                        )
                except asyncio.TimeoutError:
                    job_log(job_id, f"  {f['name']}: timeout saat fetch/vision, skip")
                    return None
                except Exception as e:
                    job_log(job_id, f"  Vision error {f['name']}: {e}")
                    return None

            vision_results = await asyncio.gather(*[fetch_vision(f) for f in batch])

            # ── Phase 2: RunPod paralel ──
            async def process_with_prefetched(f, img_and_texts):
                if img_and_texts is None:
                    row      = db_get_job(job_id)
                    failed   = json.loads(row["failed_files"] or "[]")
                    failed.append(f["name"])
                    db_update_job(job_id, failed_files=json.dumps(failed))
                    db_mark_file_processed(job_id, f["name"], "failed", 0)
                    completed = db_increment_completed(job_id)
                    if job_id in jobs:
                        jobs[job_id]["completed_files"] = completed
                    return "failed"

                return await process_image(
                    f["id"], f["name"], output_folder_id, job_id,
                    img_and_texts=img_and_texts,
                    local_output_dir=local_output_dir,
                )

            results = await asyncio.gather(*[
                process_with_prefetched(f, vr)
                for f, vr in zip(batch, vision_results)
            ])

            batch_time = time.time() - t0
            file_times.append(batch_time / max(len(batch), 1))

            for r in results:
                if r == "success":   success_count += 1
                elif r == "failed":  failed_count  += 1
                elif r == "skip":    skip_count    += 1

            if file_times:
                avg         = sum(file_times) / len(file_times)
                remaining   = len(pending_files) - (i + len(batch))
                eta_seconds = int(avg * remaining)
                if job_id in jobs:
                    jobs[job_id]["eta"] = format_eta(eta_seconds)

        # ── Finalize ──
        row_final    = db_get_job(job_id)
        failed_files = json.loads(row_final["failed_files"] or "[]")

        current_status = db_get_job(job_id)["status"]
        if current_status not in ("cancelled",):
            all_failed   = (failed_count > 0 and success_count == 0 and skip_count == 0)
            final_status = "failed" if all_failed else "completed"
            db_update_job(
                job_id,
                status=final_status,
                finished_at=datetime.now().isoformat(),
            )
            if job_id in jobs:
                jobs[job_id]["status"]      = final_status
                jobs[job_id]["finished_at"] = datetime.now().isoformat()

        if output_folder_id:
            db_schedule_deletion(output_folder_id)

        job_log(job_id,
            f"Done! ✅ {success_count} berhasil | ⏭ {skip_count} skip | ❌ {failed_count} gagal"
        )

        if local_output_dir:
            job_log(job_id, f"Output lokal: {os.path.abspath(local_output_dir)}")
        else:
            job_log(job_id, f"Output Drive: {folder_name} (ID: {output_folder_id})")

    except Exception as e:
        db_update_job(job_id, status="failed")
        if job_id in jobs:
            jobs[job_id]["status"] = "failed"
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


# ── Auto-Delete Loop ──────────────────────────────────────

async def deletion_loop():
    from core.drive import delete_folder
    from core.database import db_get_pending_deletions, db_remove_pending_deletion
    while True:
        await asyncio.sleep(60)
        try:
            for folder_id in db_get_pending_deletions():
                success = await asyncio.to_thread(delete_folder, folder_id)
                if success:
                    db_remove_pending_deletion(folder_id)
                    print(f"Auto-deleted: {folder_id}")
                else:
                    print(f"Delete failed, will retry: {folder_id}")
        except Exception as e:
            print(f"Deletion loop error: {e}")
