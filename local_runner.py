"""Local CLI runner for processing a folder of Webtoon images through RunPod."""
import argparse
import asyncio
import os
import socket
import sys
import traceback
from pathlib import Path

import aiohttp
from PIL import Image

from core.config import MAX_WIDTH
from core.local_paths import default_output_dir, discover_images
import core.runpod_client as runpod_module
from core.runpod_client import run_runpod_lama

DEFAULT_WORKERS = 3
TIMEOUT_SECONDS = 180


""Local CLI runner for processing a folder of Webtoon images through RunPod."""
import argparse
import asyncio
import os
import socket
import sys
import traceback
from pathlib import Path

import aiohttp
from PIL import Image

from core.config import MAX_WIDTH
from core.local_paths import default_output_dir, discover_images
import core.runpod_client as runpod_module
from core.runpod_client import run_runpod_lama

DEFAULT_WORKERS = 3
TIMEOUT_SECONDS = 180


def default_output_dir(input_dir: Path) -> Path:
    return input_dir.with_name(f"{input_dir.name}_cleaned")


def discover_images(input_dir: Path) -> list[Path]:
    import re
    def natural_key(path: Path):
        return [int(part) if part.isdigit() else part.casefold()
                for part in re.split(r"(\d+)", path.name)]
    return sorted(
        (p for p in input_dir.iterdir()
         if p.is_file() and p.suffix.casefold() in VALID_EXTENSIONS),
        key=natural_key,
    )


def _save_image(image: Image.Image, destination: Path) -> None:
    suffix = destination.suffix.casefold()
    if suffix in {".jpg", ".jpeg"}:
        options = {"format": "JPEG", "quality": 95, "optimize": True}
    elif suffix == ".webp":
        options = {"format": "WEBP", "quality": 95, "method": 4}
    elif suffix == ".png":
        options = {"format": "PNG", "optimize": True}
    else:
        raise ValueError(f"Unsupported output extension: {suffix}")
    image.convert("RGB").save(destination, **options)


async def process_one(source_path, output_dir, semaphore, http_session):
    async with semaphore:
        try:
            print(f"[START] {source_path.name}", flush=True)
            with Image.open(source_path) as source:
                image = source.convert("RGB")
            if image.width > MAX_WIDTH:
                ratio = MAX_WIDTH / image.width
                image = image.resize(
                    (MAX_WIDTH, max(1, int(image.height * ratio))),
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
            destination = output_dir / source_path.name
            await asyncio.to_thread(_save_image, result, destination)
            print(f"[OK] {source_path.name} -> {destination}", flush=True)
            return source_path.name, "success"
        except Exception as exc:
            print(f"[FAILED] {source_path.name}: {exc}", flush=True)
            traceback.print_exc()
            return source_path.name, "failed"


async def run_local(input_dir: Path, output_dir: Path, workers: int = DEFAULT_WORKERS):
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Output folder must be different from the input folder")
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    files = discover_images(input_dir)
    if not files:
        raise ValueError(f"No supported images found in: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input   : {input_dir}")
    print(f"Output  : {output_dir}")
    print(f"Images  : {len(files)}")
    print(f"Workers : {workers}\n")

    runpod_module.runpod_sem = asyncio.Semaphore(workers)
    timeout = aiohttp.ClientTimeout(total=300, connect=20, sock_read=300)
    connector = aiohttp.TCPConnector(
        limit=max(10, workers * 2), ttl_dns_cache=300, family=socket.AF_INET
    )
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        semaphore = asyncio.Semaphore(workers)
        results = await asyncio.gather(*[
            process_one(path, output_dir, semaphore, session) for path in files
        ])
    succeeded = sum(status == "success" for _, status in results)
    failed = len(results) - succeeded
    print(f"\n=== FINISHED: {succeeded} succeeded, {failed} failed ===")
    print(f"Output: {output_dir}")
    return succeeded, failed


def build_parser():
    parser = argparse.ArgumentParser(
        description="Clean Webtoon images in a local folder using RunPod YOLO + LaMa."
    )
    parser.add_argument("input", nargs="?", default="input",
                        help="Folder containing source images (default: ./input).")
    parser.add_argument("--output", "-o",
                        help="Output folder; default is sibling '<input>_cleaned'.")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS,
                        help=f"Maximum concurrent RunPod jobs (default: {DEFAULT_WORKERS}).")
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
