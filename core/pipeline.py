# ============================================================
# core/pipeline.py
# BASELINE — rollback dari versi overengineered
#
# process_image():
#   download → vision → smart_clean() → get_inpaint_crop()
#   → _maybe_resize_for_runpod() → RunPod LaMa
#   → validate_inpaint() → paste → upload
#
# Dihapus dari versi sebelumnya:
#   - loop for idx, (bx1,by1,bx2,by2) in enumerate(inpaint_boxes)
#   - solid_fill_inpaint() call
#   - per-ROI logging yang excessive
# ============================================================

import gc
import io
import json
import math
import time
import asyncio
import traceback
from datetime import datetime, timedelta
from typing import Optional

import aiohttp
import numpy as np
from PIL import Image, ImageFilter
from google.cloud import vision

from core.config import (
    MAX_WIDTH,
    DRIVE_OUTPUT_FOLDER_ID,
    INPAINT_CROP_PAD,
    MAX_RUNPOD_PIXELS,
    OUTPUT_DELETE_DELAY_MINUTES,
)
from core.database import (
    db_get_job, db_update_job, db_append_log, db_mark_file_processed,
    db_get_processed_files, db_schedule_deletion, db_increment_completed,
    db_remove_pending_deletion,
)
from core.drive import (
    _get_drive, filter_and_sort_files, create_output_folder,
    upload_file, download_file_async, extract_folder_id, delete_folder,
)
from core.image_processing import (
    to_bytes, progress_bar, format_eta, get_dynamic_batch_size,
    smart_clean, validate_inpaint, get_inpaint_crop,
)
from core.runpod_client import run_runpod_lama

# ── Shared State ──────────────────────────────────────────
pipeline_sem:  Optional[asyncio.Semaphore]    = None
vision_sem:    Optional[asyncio.Semaphore]    = None
_warmup_lock:  Optional[asyncio.Lock]         = None
_http_session: Optional[aiohttp.ClientSession] = None
_runpod_warmed = False

jobs           = {}
job_queue      = []
cancelled_jobs = set()
vision_client  = None

SLICE_HEIGHT        = 1500
SLICE_OVERLAP       = 200
LOG_MAX_ENTRIES     = 100
FETCH_VISION_TIMEOUT = 90
INPAINT_BLEND_BLUR  = 2   # radius kecil — cukup untuk feather edge tanpa halo


def _detect_text_sliced(image: Image.Image, vc) -> list:
    """
    Kirim gambar ke Vision API.
    Untuk gambar tinggi (webtoon), slice per SLICE_HEIGHT agar tidak
    melebihi batas ukuran Vision API.
    """
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
            xs    = [v.x for v in verts]
            ys    = [v.y + y for v in verts]

            dedup_key = (ann.description, min(xs) // 5, min(ys) // 5)
            if dedup_key in seen_texts:
                continue
            seen_texts.add(dedup_key)

            new_ann = _Ann(
                desc=ann.description,
                bp=_BP([_V(xs[i], ys[i]) for i in range(len(verts))])
            )
            all_annotations.append(new_ann)

        if y2 >= h:
            break
        y = y2 - SLICE_OVERLAP

    return all_annotations


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


# ── Resize crop sebelum kirim ke RunPod ───────────────────

def _maybe_resize_for_runpod(
    img_crop:  Image.Image,
    mask_crop: Image.Image,
) -> tuple[Image.Image, Image.Image, bool, int, int]:
    """
    Resize crop jika melebihi MAX_RUNPOD_PIXELS.
    LaMa butuh dimensi kelipatan 8.
    Returns (img, mask, was_resized, orig_w, orig_h).
    """
    orig_w, orig_h = img_crop.size
    was_resized    = False

    if orig_w * orig_h > MAX_RUNPOD_PIXELS:
        scale     = (MAX_RUNPOD_PIXELS / (orig_w * orig_h)) ** 0.5
        new_w     = max(64, (int(orig_w * scale) // 8) * 8)
        new_h     = max(64, (int(orig_h * scale) // 8) * 8)
        img_crop  = img_crop.resize((new_w, new_h), Image.LANCZOS)
        mask_crop = mask_crop.resize((new_w, new_h), Image.NEAREST)
        was_resized = True

    return img_crop, mask_crop, was_resized, orig_w, orig_h


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
            from core.config import RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID
            import base64 as _b64

            tiny_img  = Image.new("RGB", (128, 128), color=(255, 255, 255))
            tiny_mask = Image.new("L",   (128, 128), color=0)

            def _to_b64(img, fmt):
                buf = io.BytesIO()
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

            t0 = time.time()
            try:
                async with _http_session.post(
                    f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as r:
                    elapsed  = time.time() - t0
                    status   = (await r.json()).get("status", "?") if r.status == 200 else f"HTTP {r.status}"
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

        # ── Download + Vision (jika belum di-prefetch) ────
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

        # ── No text found ─────────────────────────────────
        if not texts:
            out_buf = to_bytes(img)
            await _save_output(out_buf, filename, output_folder_id, local_output_dir)
            return "skip"

        # ── Build mask ────────────────────────────────────
        img_np = np.array(img)

        prefilled, lama_mask, _, dialog_count, _ = await asyncio.to_thread(
            smart_clean, img, texts, img_np
        )
        del img_np

        # ── Inpaint ───────────────────────────────────────
        if lama_mask.getbbox() is None:
            # Mask kosong — tidak ada yang perlu di-inpaint
            final = prefilled
        else:
            # Satu crop dari seluruh bounding box mask
            crop_result = get_inpaint_crop(prefilled, lama_mask, pad=INPAINT_CROP_PAD)

            if crop_result is None:
                final = prefilled
            else:
                img_crop, mask_crop, (cx1, cy1, cx2, cy2) = crop_result

                iw, ih     = prefilled.size
                crop_area  = (cx2 - cx1) * (cy2 - cy1)
                crop_pct   = 100 * crop_area / (iw * ih)

                job_log(job_id,
                    f"  {filename}: {dialog_count} boxes | "
                    f"crop {cx2-cx1}x{cy2-cy1}px ({crop_pct:.1f}%)"
                )

                # Resize jika crop terlalu besar untuk LaMa
                img_send, mask_send, was_resized, orig_w, orig_h = \
                    _maybe_resize_for_runpod(img_crop, mask_crop)

                if was_resized:
                    job_log(job_id,
                        f"  {filename}: resize {orig_w}x{orig_h} → "
                        f"{img_send.width}x{img_send.height}"
                    )

                raw_inpaint = await run_runpod_lama(
                    img_send, mask_send,
                    label=filename,
                    http_session=_http_session,
                )

                # Upscale hasil ke ukuran crop original
                if was_resized and raw_inpaint is not None:
                    raw_inpaint = raw_inpaint.resize((orig_w, orig_h), Image.LANCZOS)

                inpaint = validate_inpaint(raw_inpaint, img_crop)

                if inpaint is not None:
                    # Blend dengan soft mask — radius kecil untuk manga hard edge
                    soft_mc    = mask_crop.filter(
                        ImageFilter.GaussianBlur(INPAINT_BLEND_BLUR)
                    )
                    final      = prefilled.copy()
                    roi_patch  = prefilled.crop((cx1, cy1, cx2, cy2))
                    roi_patch.paste(inpaint, (0, 0))
                    final.paste(roi_patch, (cx1, cy1), soft_mc)
                else:
                    job_log(job_id, f"  {filename}: inpaint corrupt, pakai prefilled")
                    final = prefilled

        # ── Save ──────────────────────────────────────────
        out_buf = to_bytes(final)
        await _save_output(out_buf, filename, output_folder_id, local_output_dir)

        duration  = time.time() - start_time
        completed = db_increment_completed(job_id)
        if job_id in jobs:
            jobs[job_id]["completed_files"] = completed

        db_mark_file_processed(job_id, filename, "success", duration)

        row = db_get_job(job_id)
        job_log(job_id,
            f"{progress_bar(completed, row['total_files'] or 0)} "
            f"{filename} ({duration:.1f}s)"
        )
        return "success"

    except Exception as e:
        duration  = time.time() - start_time
        completed = db_increment_completed(job_id)
        if job_id in jobs:
            jobs[job_id]["completed_files"] = completed

        row    = db_get_job(job_id)
        failed = json.loads(row["failed_files"] or "[]")
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
    import os
    if local_output_dir:
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

            # Prefetch: download + vision paralel untuk seluruh batch
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

            vision_results = await asyncio.gather(*[fetch_vision(f) for f in batch])

            # Proses inpaint per file dengan hasil prefetch
            async def process_with_prefetched(f, img_and_texts):
                if img_and_texts is None:
                    row    = db_get_job(job_id)
                    failed = json.loads(row["failed_files"] or "[]")
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
                )

            results = await asyncio.gather(*[
                process_with_prefetched(f, vr)
                for f, vr in zip(batch, vision_results)
            ])

            batch_time = time.time() - t0
            file_times.append(batch_time / max(len(batch), 1))

            for r in results:
                if r == "success":  success_count += 1
                elif r == "failed": failed_count  += 1
                elif r == "skip":   skip_count    += 1

            if file_times:
                avg         = sum(file_times) / len(file_times)
                remaining   = len(pending_files) - (i + len(batch))
                eta_seconds = int(avg * remaining)
                if job_id in jobs:
                    jobs[job_id]["eta"] = format_eta(eta_seconds)

        # ── Finalize ──────────────────────────────────────
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
            delete_at = datetime.utcnow() + timedelta(minutes=OUTPUT_DELETE_DELAY_MINUTES)
            unix_ts   = int(delete_at.timestamp())
            if job_id in jobs:
                jobs[job_id]["delete_at_unix"] = unix_ts

        job_log(job_id,
            f"Done! ✅ {success_count} berhasil | ⏭ {skip_count} skip | ❌ {failed_count} gagal"
        )
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

    # Force return unused heap ke OS setelah job selesai
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
