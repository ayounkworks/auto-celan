# ============================================================
# local_runner.py
# Mode lokal: python main.py "<folder chapter>" [--workers N]
# ============================================================

import argparse
import asyncio
import re
import traceback
import socket
from datetime import datetime
from pathlib import Path

import aiohttp
from PIL import Image

from core.config import MAX_WIDTH
import core.runpod_client as runpod_module
from core.runpod_client import run_runpod_lama

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "input"
DEFAULT_OUTPUT = PROJECT_DIR / "output"
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_WORKERS = 10
TIMEOUT = 180


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("workers harus berupa angka bulat.")
    if parsed < 1 or parsed > 64:
        raise argparse.ArgumentTypeError("workers harus berada di rentang 1–64.")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="autoclean",
        description="Bersihkan semua gambar dalam folder chapter menggunakan Auto Celan.",
    )
    parser.add_argument(
        "folder", nargs="?", default=str(DEFAULT_INPUT),
        help="Folder chapter berisi gambar. Default: folder input/ proyek.",
    )
    parser.add_argument(
        "-w", "--workers", type=positive_int, default=DEFAULT_WORKERS,
        help=f"Jumlah gambar yang diproses bersamaan (default: {DEFAULT_WORKERS}, maksimum: 64).",
    )
    parser.add_argument(
        "-o", "--output", default=str(DEFAULT_OUTPUT),
        help="Folder induk output (default: output/ proyek).",
    )
    return parser.parse_args(argv)


async def process_one(img_path: Path, sem, http_session, output_dir: Path):
    async with sem:
        try:
            print(f"[START] {img_path.name}")
            with Image.open(img_path) as source:
                img = source.convert("RGB")

            if img.width > MAX_WIDTH:
                ratio = MAX_WIDTH / img.width
                img = img.resize(
                    (MAX_WIDTH, int(img.height * ratio)),
                    Image.Resampling.LANCZOS,
                )

            try:
                result = await asyncio.wait_for(
                    run_runpod_lama(img, None, label=img_path.name, http_session=http_session),
                    timeout=TIMEOUT,
                )
            except asyncio.TimeoutError:
                print(f"[TIMEOUT] {img_path.name}, menyimpan gambar asli sebagai fallback.")
                img.save(output_dir / img_path.name, quality=95)
                return

            if result is None:
                print(f"[FAILED] RunPod mengembalikan None untuk {img_path.name}; menyimpan gambar asli.")
                img.save(output_dir / img_path.name, quality=95)
                return

            result.save(output_dir / img_path.name, quality=95)
            print(f"[OK] {img_path.name}")

        except Exception as exc:
            print(f"[ERROR] {img_path.name}: {exc}")
            traceback.print_exc()


async def run_local(input_dir: Path, output_root: Path, workers: int):
    if not input_dir.exists():
        raise FileNotFoundError(f"Folder input tidak ditemukan: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Path bukan folder: {input_dir}")

    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXT
    )
    if not files:
        print(f"Tidak ada gambar (.jpg, .jpeg, .png, .webp) di: {input_dir}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", input_dir.name).strip("._") or "chapter"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_output = output_root / f"{safe_name}_{timestamp}"
    job_output.mkdir(parents=True, exist_ok=False)

    timeout = aiohttp.ClientTimeout(total=300, connect=20, sock_read=300)
    connector = aiohttp.TCPConnector(limit=max(workers, 10), ttl_dns_cache=300, family=socket.AF_INET)

    print(f"Input   : {input_dir}")
    print(f"Output  : {job_output}")
    print(f"Images  : {len(files)}")
    print(f"Workers : {workers}\n")

    runpod_module.runpod_sem = asyncio.Semaphore(workers)
    semaphore = asyncio.Semaphore(workers)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http_session:
        tasks = [process_one(path, semaphore, http_session, job_output) for path in files]
        await asyncio.gather(*tasks)

    print("\n=== PROSES SELESAI ===")
    print(f"Hasil tersimpan di: {job_output}")


def main(argv=None):
    # No CLI arguments: use an interactive prompt for a simpler workflow.
    interactive = argv is None and len(__import__("sys").argv) == 1

    if interactive:
        print("=" * 38)
        print("             AUTO CELAN")
        print("=" * 38)
        folder = input("\nMasukkan path folder chapter:\n> ").strip().strip('"').strip("'")
        if not folder:
            raise SystemExit("Path folder tidak boleh kosong.")

        workers = DEFAULT_WORKERS
        args = argparse.Namespace(folder=folder, workers=workers, output=str(DEFAULT_OUTPUT))
    else:
        args = parse_args(argv)

    input_dir = Path(args.folder).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    print("\n=== Auto Celan — Local Mode ===")
    try:
        asyncio.run(run_local(input_dir, output_root, args.workers))
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
