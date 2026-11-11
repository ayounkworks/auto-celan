""""Standalone CLI: local files in/out, RunPod used only for GPU inference."""
import argparse
import asyncio
import socket
import sys
import traceback
from pathlib import Path

import aiohttp
from PIL import Image, UnidentifiedImageError

from core.config import MAX_WIDTH, RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID
from core.local_paths import default_output_dir, discover_images
import core.runpod_client as runpod_module
from core.runpod_client import run_runpod_lama

DEFAULT_WORKERS = 2
TIMEOUT_SECONDS = 300


def _save_image(image: Image.Image, destination: Path) -> None:
    """Save using the source extension with explicit format/quality."""
    suffix = destination.suffix.casefold()
    if suffix in {".jpg", ".jpeg"}:
        image.convert("RGB").save(destination, format="JPEG", quality=95, optimize=True)
    elif suffix == ".webp":
        image.convert("RGB").save(destination, format="WEBP", quality=95, method=4)
    elif suffix == ".png":
        image.convert("RGB").save(destination, format="PNG", optimize=True)
    else:
        raise ValueError(f"Unsupported output extension: {suffix}")


async def process_one(source_path: Path, output_dir: Path, semaphore, http_session):
    async with semaphore:
        print(f"[START] {source_path.name}", flush=True)
        try:
            with Image.open(source_path) as source:
                source.load()
                original_size = source.size
                image = source.convert("RGB")

            if image.width > MAX_WIDTH:
                ratio = MAX_WIDTH / image.width
                image = image.resize(
                    (MAX_WIDTH, max(1, round(image.height * ratio))),
                    Image.Resampling.LANCZOS,
                )

            result = await asyncio.wait_for(
                run_runpod_lama(
                    image, None, label=source_path.name,
                    http_session=http_session, filename=source_path.name,
                ),
                timeout=TIMEOUT_SECONDS,
            )
            if result is None:
                raise RuntimeError("RunPod returned no image")

            if result.size != original_size:
                result = result.resize(original_size, Image.Resampling.LANCZOS)

            destination = output_dir / source_path.name
            await asyncio.to_thread(_save_image, result, destination)
            print(f"[OK] {source_path.name} -> {destination}", flush=True)
            return source_path.name, True
        except Exception as exc:
            print(f"[FAILED] {source_path.name}: {exc}", flush=True)
            traceback.print_exc()
            return source_path.name, False


async def run_local(input_dir: Path, output_dir: Path, workers: int = DEFAULT_WORKERS):
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not RUNPOD_API_KEY:
        raise ValueError("RUNPOD_API_KEY belum diatur. Tambahkan ke file .env.")
    if not RUNPOD_ENDPOINT_ID:
        raise ValueError("RUNPOD_ENDPOINT_ID belum diatur.")
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Folder input tidak ditemukan: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Folder output tidak boleh sama dengan folder input.")
    if output_dir in input_dir.parents:
        raise ValueError("Folder output tidak boleh menjadi parent folder input.")
    if workers < 1:
        raise ValueError("--workers harus minimal 1.")

    files = discover_images(input_dir)
    if not files:
        raise ValueError(f"Tidak ada gambar JPG, JPEG, PNG, atau WEBP di: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input   : {input_dir}")
    print(f"Output  : {output_dir}")
    print(f"Images  : {len(files)}")
    print(f"Workers : {workers}")
    print("Backend : RunPod Serverless (YOLO + LaMa)\\n")

    runpod_module.runpod_sem = asyncio.Semaphore(workers)
    timeout = aiohttp.ClientTimeout(total=360, connect=30, sock_read=300)
    connector = aiohttp.TCPConnector(
        limit=max(4, workers * 2), ttl_dns_cache=300, family=socket.AF_INET
    )

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        semaphore = asyncio.Semaphore(workers)
        results = await asyncio.gather(*[
            process_one(path, output_dir, semaphore, session) for path in files
        ])

    succeeded = sum(ok for _, ok in results)
    failed = len(results) - succeeded
    print(f"\\n=== SELESAI: {succeeded} berhasil, {failed} gagal ===")
    print(f"Output: {output_dir}")
    return succeeded, failed


def build_parser():
    parser = argparse.ArgumentParser(
        prog="auto-celan",
        description="Bersihkan halaman Webtoon dari folder lokal. Inferensi GPU menggunakan RunPod.",
    )
    parser.add_argument("input", nargs="?", default="input",
                        help="Folder gambar sumber (default: ./input).")
    parser.add_argument("--output", "-o",
                        help="Folder hasil. Default: subfolder output_cleaned di dalam folder input.")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS,
                        help=f"Jumlah gambar paralel (default: {DEFAULT_WORKERS}).")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else default_output_dir(input_dir)
    try:
        _, failed = asyncio.run(run_local(input_dir, output_dir, args.workers))
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
