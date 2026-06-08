# ============================================================
# core/pipeline.py
# FIXED v4:
# - BUG_AUTODELETE: asyncio.create_task di loop yang salah
#   Pipeline jalan di loop baru (thread), auto-delete harus
#   dijadwalkan di main bot loop via schedule_folder_delete()
# - BUG3: Closure bug fetch_vision (default arg)
# - BUG5: Hapus input_folder_name tidak terpakai
# - BUG7: Limit log 100 entry
# - INPAINT: Tambah db_schedule_deletion sebagai backup delete
# ============================================================

import gc
import io
import json
import time
import asyncio
import traceback
from datetime import datetime, timedelta
from typing import Optional

import aiohttp
import numpy as np
from PIL import Image, ImageFilter
from google.cloud import vision

from core.config import MAX_WIDTH, DRIVE_OUTPUT_FOLDER_ID, INPAINT_CROP_PAD
from core.database import (
    db_get_job, db_update_job, db_append_log, db_mark_file_processed,
    db_get_processed_files, db_schedule_deletion, db_increment_completed,
    db_remove_pending_deletion,
)
from core.drive import (
    _get_drive, CREDS, filter_and_sort_files, create_output_folder,
    upload_file, download_file_async, extract_folder_id, delete_folder,
)
from core.image_processing import (
    to_bytes, progress_bar, format_eta, get_dynamic_batch_size,
    smart_clean, validate_inpaint,
)
from core.runpod_client import run_runpod_lama

# ── Shared State ──────────────────────────────────────────
pipeline_sem:  Optional[asyncio.Semaphore] = None
vision_sem:    Optional[asyncio.Semaphore] = None
_warmup_lock:  Optional[asyncio.Lock]      = None
_http_session: Optional[aiohttp.ClientSession] = None
_runpod_warmed = False

jobs           = {}

SLICE_HEIGHT   = 1500
SLICE_OVERLAP  = 200

OUTPUT_DELETE_DELAY_MINUTES = 15

LOG_MAX_ENTRIES = 100


def _detect_text_sliced(image: Image.Image, vc) -> list:
    w, h = image.size

    if h <= int(SLICE_HEIGHT * 1.5):
        buf = to_bytes(image, "JPEG", 95)
        buf.seek(0)
        resp = vc.document_text_detection(image=vision.Image(content=buf.read()))
        return list(resp.text_annotations)

    class _V:
        __slots__ = ("x", "y")
        def __init__(self, x, y): self.x = x; self.y = y

    class _BP:
        __slots__ = ("vertices",)
        def __init__(self, verts): self.vertices = verts

    class _Ann:
        __slots__ = ("description", "bounding_poly")
        def __init__(self, desc, bp): self.description = desc; self.bounding_poly = bp

    all_annotations = []
    seen_texts      = set()

    y = 0
    while y < h:
        y2        = min(y + SLICE_HEIGHT, h)
        slice_img = image.crop((0, y, w, y2))

        buf = to_bytes(slice_img, "JPEG", 95)
        buf.seek(0)
        try:
            resp = vc.document_text_detection(image=vision.Image(content=buf.read()))
        except Exception as e:
            print(f"[slice_vision] Error at y={y}-{y2}: {e}")
            if y2 >= h: break
            y = y2 - SLICE_OVERLAP
            continue

        annotations = resp.text_annotations
        if not annotations:
            if y2 >= h: break
            y = y2 - SLICE_OVERLAP
            continue

        for ann in annotations[1:]:
            verts = ann.bounding_poly.vertices
            xs    = [v.x       for v in verts]
            ys    = [v.y + y   for v in verts]

            # Gunakan resolusi dedup yang lebih tinggi (5px) agar teks rapat tidak hilang
            dedup_key = (ann.description, min(xs) // 5, min(ys) // 5)
            if dedup_key in seen_texts:
                continue
            seen_texts.add(dedup_key)

            new_ann = _Ann(
                desc = ann.description,
                bp   = _BP([_V(xs[i], ys[i]) for i in range(len(verts))])
            )
            all_annotations.append(new_ann)

        if y2 >= h:
            break
        y = y2 - SLICE_OVERLAP

    return all_annotations


job_queue      = []
cancelled_jobs = set()
vision_client  = None

FETCH_VISION_TIMEOUT = 90


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

        msg = "🔥 Warming up RunPod worker..."
        print(msg)
        if job_id:
            job_log(job_id, msg)

        try:
            from PIL import Image as _Image
            import io as _io
            import base64 as _b64
            from core.config import RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID

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
            warn = f"⚠️ Warmup skip: {e}"
            print(warn)
            if job_id:
                job_log(job_id, warn)

        finally:
            _runpod_warmed = True


# ── Auto-Delete ───────────────────────────────────────────

async def _auto_delete_output(folder_id: str, folder_name: str,
                               delay_minutes: int = OUTPUT_DELETE_DELAY_MINUTES):
    """
    Hapus output folder setelah delay_minutes.
    Dipanggil dari deletion_loop() di main bot event loop.
    """
    print(f"⏰ Output '{folder_name}' akan dihapus dalam {delay_minutes} menit")
    await asyncio.sleep(delay_minutes * 60)
    try:
        success = await asyncio.to_thread(delete_folder, folder_id)
        if success:
            db_remove_pending_deletion(folder_id)
            print(f"🗑️ Auto-deleted output: {folder_name} ({folder_id})")
        else:
            print(f"❌ Gagal hapus output: {folder_name} ({folder_id})")
    except Exception as e:
        print(f"❌ Auto-delete error: {e}")


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

        if img_and_texts is None:
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
                    texts = _detect_text_sliced(image, vision_client)
                    return image, texts

                img, texts = await asyncio.to_thread(resize_and_detect)
        else:
            img, texts = img_and_texts

        if not texts:
            out_buf = to_bytes(img)
            await _save_output(out_buf, filename, output_folder_id, local_output_dir)
            status = "skip"

        else:
            img_np = np.array(img)

            prefilled, lama_mask, sfx_count, dialog_count, inpaint_boxes = await asyncio.to_thread(
                smart_clean, img, texts, img_np
            )
            del img_np

            has_mask = lama_mask.getbbox() is not None

            if has_mask:
                iw, ih = prefilled.size
                final     = prefilled.copy()
                job_log(job_id, f"  {filename}: Memproses {len(inpaint_boxes)} ROI Inpaint.")

                for idx, (bx1, by1, bx2, by2) in enumerate(inpaint_boxes):
                    # Gunakan padding untuk konteks inpainting
                    PAD = 40
                    cx1 = max(0,  bx1 - PAD)
                    cy1 = max(0,  by1 - PAD)
                    cx2 = min(iw, bx2 + PAD)
                    cy2 = min(ih, by2 + PAD)

                    img_crop  = prefilled.crop((cx1, cy1, cx2, cy2))
                    mask_crop = lama_mask.crop((cx1, cy1, cx2, cy2))

                    if mask_crop.getextrema() == (0, 0):
                        continue

                    raw_inpaint = await run_runpod_lama(
                        img_crop, mask_crop,
                        label=f"{filename}_roi{idx}",
                        http_session=_http_session,
                    )
                    inpaint = validate_inpaint(raw_inpaint, img_crop)
                    if inpaint is not None:
                        # Gunakan soft mask untuk blending halus
                        blur_r  = 5
                        soft_mc = mask_crop.filter(ImageFilter.GaussianBlur(blur_r))
                        
                        roi_result = prefilled.crop((cx1, cy1, cx2, cy2))
                        roi_result.paste(inpaint, (0, 0))
                        
                        # Paste kembali menggunakan soft mask di koordinat crop semula
                        final.paste(roi_result, (cx1, cy1), soft_mc)
                    else:
                        job_log(job_id, f"  {filename}_roi{idx}: inpaint corrupt, skip")
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

        folder_name, output_folder_id = await asyncio.to_thread(
            create_output_folder, DRIVE_OUTPUT_FOLDER_ID, "output"
        )
        local_output_dir = None

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

            async def fetch_vision(f, _f=None):
                f = _f if _f is not None else f
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
                            raw.close()
                            texts = _detect_text_sliced(image, vision_client)
                            return image, texts

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

            vision_results = await asyncio.gather(*[fetch_vision(f, _f=f) for f in batch])

            async def process_with_prefetched(f, img_and_texts, _f=None, _vr=None):
                f             = _f if _f is not None else f
                img_and_texts = _vr if _vr is not None else img_and_texts

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
                process_with_prefetched(f, vr, _f=f, _vr=vr)
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

        # ── FIXED AUTO-DELETE: Simpan ke DB, deletion_loop di main bot loop yang eksekusi ──
        if output_folder_id:
            db_schedule_deletion(output_folder_id)
            delete_at = datetime.utcnow() + timedelta(minutes=OUTPUT_DELETE_DELAY_MINUTES)
            unix_ts   = int(delete_at.timestamp())
            if job_id in jobs:
                jobs[job_id]["delete_at_unix"] = unix_ts

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


# ── Auto-Delete Loop (jalan di main bot event loop) ───────
# FIXED: deletion_loop berjalan di event loop bot (main),
# bukan di thread pipeline. Ini yang memastikan delete terjadi.

async def deletion_loop():
    from core.database import db_get_pending_deletions, db_remove_pending_deletion
    while True:
        await asyncio.sleep(60)  # cek setiap menit
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