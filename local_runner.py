# ============================================================
# local_runner.py
# Mode lokal: python main.py local
#
# FIXES:
# 1. vision_client diinit setelah .env di-load (tidak di module level)
# 2. asyncio.wait_for() per file → tidak hang selamanya
# 3. Semaphore runpod dan concurrency lebih seimbang
# ============================================================

import asyncio
import os
import traceback
import socket
from datetime import datetime
from io import BytesIO

import aiohttp
import numpy as np
from PIL import Image, ImageFilter
from google.cloud import vision as gvision

from core.config import GOOGLE_API_KEY, MAX_WIDTH
from core.image_processing import (
    to_bytes, smart_clean,
    validate_inpaint,
)

import core.runpod_client as runpod_module
from core.runpod_client import run_runpod_lama


FOLDER_TARGET = os.path.join(os.path.dirname(__file__), "input")
FOLDER_OUTPUT = os.path.join(os.path.dirname(__file__), "output")

VALID_EXT     = (".jpg", ".jpeg", ".png", ".webp")
CONCURRENCY   = 5    # file paralel
FETCH_TIMEOUT = 90   # detik per file


async def process_one(
    filename:     str,
    sem:          asyncio.Semaphore,
    http_session: aiohttp.ClientSession,
    output_dir:   str,
    vision_client,
):
    async with sem:
        try:
            print(f"[START] {filename}")
            img_path = os.path.join(FOLDER_TARGET, filename)

            # Load + resize
            img = Image.open(img_path).convert("RGB")
            if img.width > MAX_WIDTH:
                ratio = MAX_WIDTH / img.width
                img   = img.resize(
                    (MAX_WIDTH, int(img.height * ratio)),
                    Image.Resampling.LANCZOS,
                )

            buf = to_bytes(img, "JPEG", 95)
            buf.seek(0)
            vision_image = gvision.Image(content=buf.read())

            loop = asyncio.get_event_loop()

            # Vision API dengan timeout
            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: vision_client.document_text_detection(image=vision_image),
                    ),
                    timeout=FETCH_TIMEOUT,
                )
            except asyncio.TimeoutError:
                print(f"[TIMEOUT] Vision API timeout untuk {filename}, skip")
                img.save(os.path.join(output_dir, filename))
                return

            texts = response.text_annotations

            if not texts:
                print(f"[SKIP] Tidak ada teks → {filename}")
                out_path = os.path.join(output_dir, filename)
                await loop.run_in_executor(None, lambda: img.save(out_path))
                return

            img_np = np.array(img)

            prefilled, lama_mask, sfx_c, dial_c = await loop.run_in_executor(
                None, smart_clean, img, texts, img_np,
            )

            print(f"  {filename}: {dial_c} dialog, {sfx_c} SFX")

            has_mask = lama_mask.getbbox() is not None

            if has_mask:
                try:
                    raw_inpaint = await asyncio.wait_for(
                        run_runpod_lama(
                            prefilled, lama_mask,
                            label=filename,
                            http_session=http_session,
                        ),
                        timeout=120,
                    )
                except asyncio.TimeoutError:
                    print(f"[TIMEOUT] RunPod timeout untuk {filename}, skip")
                    prefilled.save(os.path.join(output_dir, filename), quality=95)
                    return

                inpaint = validate_inpaint(raw_inpaint, prefilled)
                if inpaint is not None:
                    soft_mask = lama_mask.filter(ImageFilter.GaussianBlur(7))
                    final     = prefilled.copy()
                    final.paste(inpaint, (0, 0), soft_mask)
                else:
                    raise Exception("Inpaint result corrupt, file skipped")
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

    files = [
        f for f in os.listdir(FOLDER_TARGET)
        if f.lower().endswith(VALID_EXT)
    ]

    if not files:
        print("❌ Tidak ada gambar di folder input/")
        return

    # FIX: inisialisasi vision_client DI SINI (setelah .env sudah di-load)
    vision_client = gvision.ImageAnnotatorClient(
        client_options={"api_key": GOOGLE_API_KEY}
    )

    timeout   = aiohttp.ClientTimeout(total=300, connect=20, sock_connect=20, sock_read=300)
    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300, family=socket.AF_INET)

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_output = os.path.join(FOLDER_OUTPUT, f"run_{ts}")
    os.makedirs(job_output, exist_ok=True)

    print(f"Membaca dari : {FOLDER_TARGET}")
    print(f"Output ke   : {job_output}")
    print(f"Total file  : {len(files)}\n")

    runpod_module.runpod_sem = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http_session:
        sem   = asyncio.Semaphore(CONCURRENCY)
        tasks = [
            process_one(f, sem, http_session, job_output, vision_client)
            for f in sorted(files)
        ]
        await asyncio.gather(*tasks)

    print("\n=== SEMUA SELESAI ===")


def main():
    print("=== Mode Lokal ===")
    asyncio.run(run_local())
