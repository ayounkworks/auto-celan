# ============================================================
# local_runner.py
# Mode lokal: python main.py local
#
# Flow: baca dari input/ → kirim ke RunPod (YOLO+LaMa) → simpan ke output/
# Tidak ada YOLO di sini — semua di RunPod
# ============================================================

import asyncio
import os
import traceback
import socket
from datetime import datetime

import aiohttp
from PIL import Image

from core.config import MAX_WIDTH
from core.image_processing import to_bytes
import core.runpod_client as runpod_module
from core.runpod_client import run_runpod_lama

FOLDER_TARGET = os.path.join(os.path.dirname(__file__), "input")
FOLDER_OUTPUT = os.path.join(os.path.dirname(__file__), "output")

VALID_EXT   = (".jpg", ".jpeg", ".png", ".webp")
CONCURRENCY = 3
TIMEOUT     = 180


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

            # Kirim ke RunPod — YOLO detect + LaMa inpaint sekalian di sana
            try:
                result = await asyncio.wait_for(
                    run_runpod_lama(img, None, label=filename, http_session=http_session),
                    timeout=TIMEOUT,
                )
            except asyncio.TimeoutError:
                print(f"[TIMEOUT] {filename}, skip")
                img.save(os.path.join(output_dir, filename), quality=95)
                return

            if result is None:
                print(f"[FAILED] RunPod return None untuk {filename}, simpan original")
                img.save(os.path.join(output_dir, filename), quality=95)
                return

            out_path = os.path.join(output_dir, filename)
            result.save(out_path, quality=95)
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
    print("=== Mode Lokal ===")
    asyncio.run(run_local())