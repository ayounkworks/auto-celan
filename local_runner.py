# ============================================================
# local_runner.py
# Mode lokal: python main.py local
# YOLO edition — Vision API dihapus
# ============================================================

import asyncio
import os
import traceback
import socket
from datetime import datetime

import aiohttp
import numpy as np
from PIL import Image, ImageFilter

from core.config import MAX_WIDTH, YOLO_MODEL_PATH
from core.image_processing import to_bytes, smart_clean, validate_inpaint, get_inpaint_crop
from core.yolo_client import detect_text
import core.runpod_client as runpod_module
from core.runpod_client import run_runpod_lama

FOLDER_TARGET = os.path.join(os.path.dirname(__file__), "input")
FOLDER_OUTPUT = os.path.join(os.path.dirname(__file__), "output")

VALID_EXT   = (".jpg", ".jpeg", ".png", ".webp")
CONCURRENCY = 5
TIMEOUT     = 120


async def process_one(filename, sem, http_session, output_dir):
    async with sem:
        try:
            print(f"[START] {filename}")
            img_path = os.path.join(FOLDER_TARGET, filename)

            img = Image.open(img_path).convert("RGB")
            if img.width > MAX_WIDTH:
                ratio = MAX_WIDTH / img.width
                img   = img.resize(
                    (MAX_WIDTH, int(img.height * ratio)), Image.Resampling.LANCZOS
                )

            loop = asyncio.get_event_loop()

            # YOLO detection (sync, di thread pool)
            try:
                texts = await asyncio.wait_for(
                    loop.run_in_executor(None, detect_text, img, YOLO_MODEL_PATH),
                    timeout=TIMEOUT,
                )
            except asyncio.TimeoutError:
                print(f"[TIMEOUT] YOLO timeout untuk {filename}, skip")
                img.save(os.path.join(output_dir, filename))
                return

            if not texts:
                print(f"[SKIP] Tidak ada teks → {filename}")
                img.save(os.path.join(output_dir, filename))
                return

            img_np = np.array(img)
            prefilled, lama_mask, _, dialog_count, _ = await loop.run_in_executor(
                None, smart_clean, img, texts, img_np
            )
            print(f"  {filename}: {dialog_count} boxes")

            if lama_mask.getbbox() is None:
                final = prefilled
            else:
                from core.image_processing import get_inpaint_crop, INPAINT_CROP_PAD
                from core.config import MAX_RUNPOD_PIXELS
                import math

                crop_result = get_inpaint_crop(prefilled, lama_mask)
                if crop_result is None:
                    final = prefilled
                else:
                    img_crop, mask_crop, (cx1, cy1, cx2, cy2) = crop_result

                    # Simpan mask_crop original (ukuran crop) untuk paste nanti
                    orig_mask_crop = mask_crop.copy()

                    # Resize jika terlalu besar untuk RunPod
                    orig_w, orig_h = img_crop.size
                    was_resized = False
                    if orig_w * orig_h > MAX_RUNPOD_PIXELS:
                        scale = (MAX_RUNPOD_PIXELS / (orig_w * orig_h)) ** 0.5
                        nw = max(64, (int(orig_w * scale) // 8) * 8)
                        nh = max(64, (int(orig_h * scale) // 8) * 8)
                        img_crop  = img_crop.resize((nw, nh), Image.LANCZOS)
                        mask_crop = mask_crop.resize((nw, nh), Image.NEAREST)
                        was_resized = True

                    try:
                        raw_inpaint = await asyncio.wait_for(
                            run_runpod_lama(img_crop, mask_crop,
                                            label=filename, http_session=http_session),
                            timeout=TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        print(f"[TIMEOUT] RunPod timeout untuk {filename}")
                        prefilled.save(os.path.join(output_dir, filename), quality=95)
                        return

                    # Upscale hasil ke ukuran crop original
                    if was_resized and raw_inpaint is not None:
                        raw_inpaint = raw_inpaint.resize((orig_w, orig_h), Image.LANCZOS)

                    # Validate pakai ukuran crop original
                    ref_crop = Image.new("RGB", (orig_w, orig_h)) if was_resized else img_crop
                    inpaint  = validate_inpaint(raw_inpaint, ref_crop)

                    if inpaint is not None:
                        # Semua ukuran harus sama: orig_w x orig_h
                        soft_mc   = orig_mask_crop.filter(ImageFilter.GaussianBlur(2))
                        roi_patch = prefilled.crop((cx1, cy1, cx2, cy2))
                        roi_patch.paste(inpaint, (0, 0))
                        final     = prefilled.copy()
                        final.paste(roi_patch, (cx1, cy1), soft_mc)
                    else:
                        final = prefilled

            out_path = os.path.join(output_dir, filename)
            await loop.run_in_executor(None, lambda: final.save(out_path, quality=95))
            print(f"✅ Selesai → {out_path}")

        except Exception as e:
            print(f"❌ Error {filename}: {e}")
            traceback.print_exc()


async def run_local():
    os.makedirs(FOLDER_OUTPUT, exist_ok=True)

    files = [f for f in os.listdir(FOLDER_TARGET) if f.lower().endswith(VALID_EXT)]
    if not files:
        print("❌ Tidak ada gambar di folder input/")
        return

    timeout   = aiohttp.ClientTimeout(total=300, connect=20, sock_read=300)
    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300, family=socket.AF_INET)

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_output = os.path.join(FOLDER_OUTPUT, f"run_{ts}")
    os.makedirs(job_output, exist_ok=True)

    print(f"Model  : {YOLO_MODEL_PATH}")
    print(f"Input  : {FOLDER_TARGET}")
    print(f"Output : {job_output}")
    print(f"Files  : {len(files)}\n")

    runpod_module.runpod_sem = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http_session:
        sem   = asyncio.Semaphore(CONCURRENCY)
        tasks = [
            process_one(f, sem, http_session, job_output)
            for f in sorted(files)
        ]
        await asyncio.gather(*tasks)

    print("\n=== SEMUA SELESAI ===")


def main():
    print("=== Mode Lokal (YOLO) ===")
    asyncio.run(run_local())