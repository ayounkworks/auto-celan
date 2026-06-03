import runpod
import base64
import os
import sys
import time
import io
from PIL import Image, ImageDraw
from dotenv import load_dotenv

# Load .env dari parent folder (auto_celan_v2/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

runpod.api_key     = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID        = os.getenv("RUNPOD_ENDPOINT_ID")

# ============================================================
# HELPERS
# ============================================================

def image_to_base64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def base64_to_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data))

def save_result(img: Image.Image, filename: str):
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    img.save(out_path)
    print(f"Saved → {out_path}")

# ============================================================
# CORE — kirim request ke RunPod dan tunggu hasil
# ============================================================

def run_inpaint(image: Image.Image, mask: Image.Image, label="result") -> Image.Image | None:
    if not ENDPOINT_ID:
        print("ERROR: RUNPOD_ENDPOINT_ID belum diset di .env")
        return None

    print(f"\n{'='*50}")
    print(f"Job      : {label}")
    print(f"Image    : {image.size} {image.mode}")
    print(f"Mask     : {mask.size} {mask.mode}")

    image_b64 = image_to_base64(image)
    mask_b64  = image_to_base64(mask.convert("L"))

    print("Sending to RunPod...")
    start    = time.time()
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    run      = endpoint.run({"img": image_b64, "mask": mask_b64})

    print(f"Job ID   : {run.job_id}")
    print("Waiting  ", end="", flush=True)

    while True:
        status = run.status()
        print(".", end="", flush=True)

        if status == "COMPLETED":
            break
        elif status == "FAILED":
            print(f"\nFAILED: {run.output()}")
            return None

        time.sleep(1)

    elapsed = time.time() - start
    print(f"\nDone in  : {elapsed:.1f}s")

    output = run.output()

    if "error" in output:
        print(f"Handler error: {output['error']}")
        return None

    return base64_to_image(output["image"])

# ============================================================
# TEST MODES
# ============================================================

def test_dummy():
    """
    Mode 1 — Quick test tanpa file apapun.
    Buat dummy image 512x512 dengan kotak hitam di tengah sebagai mask.
    Cocok untuk verifikasi endpoint berjalan sebelum pakai gambar asli.
    """
    print("\n[MODE] Dummy test 512x512")

    img  = Image.new("RGB", (512, 512), color=(200, 200, 200))
    mask = Image.new("L",   (512, 512), color=0)

    draw = ImageDraw.Draw(mask)
    draw.ellipse([150, 180, 360, 330], fill=255)  # oval seperti speech bubble

    result = run_inpaint(img, mask, label="dummy")
    if result:
        save_result(result, "dummy_output.png")
        img.save(os.path.join(os.path.dirname(__file__), "output", "dummy_input.png"))
        mask.save(os.path.join(os.path.dirname(__file__), "output", "dummy_mask.png"))
        print("Saved input, mask, dan output untuk perbandingan visual")


def test_single(image_path: str, mask_path: str):
    """
    Mode 2 — Test dengan file gambar asli.
    Taruh gambar di test/input/ dan mask di test/mask/
    """
    print(f"\n[MODE] Single file test")

    if not os.path.exists(image_path):
        print(f"ERROR: File tidak ditemukan → {image_path}")
        return
    if not os.path.exists(mask_path):
        print(f"ERROR: Mask tidak ditemukan → {mask_path}")
        return

    img    = Image.open(image_path).convert("RGB")
    mask   = Image.open(mask_path).convert("L")
    label  = os.path.basename(image_path)
    result = run_inpaint(img, mask, label=label)

    if result:
        save_result(result, f"result_{label}")


def test_batch():
    """
    Mode 3 — Test semua gambar di folder test/input/
    Setiap gambar harus punya mask dengan nama sama di test/mask/
    """
    print(f"\n[MODE] Batch test")

    input_dir = os.path.join(os.path.dirname(__file__), "input")
    mask_dir  = os.path.join(os.path.dirname(__file__), "mask")
    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}

    files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f.lower())[1] in valid_ext
    ])

    if not files:
        print(f"Tidak ada gambar di {input_dir}")
        return

    print(f"Found {len(files)} file(s)")
    total_start = time.time()
    success     = 0

    for i, filename in enumerate(files, 1):
        image_path = os.path.join(input_dir, filename)
        mask_path  = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            print(f"[{i}/{len(files)}] SKIP {filename} — mask tidak ditemukan")
            continue

        print(f"\n[{i}/{len(files)}]", end="")
        img    = Image.open(image_path).convert("RGB")
        mask   = Image.open(mask_path).convert("L")
        result = run_inpaint(img, mask, label=filename)

        if result:
            save_result(result, f"result_{filename}")
            success += 1

    total = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"Batch selesai : {success}/{len(files)} berhasil")
    print(f"Total waktu   : {total:.1f}s")
    if success > 0:
        print(f"Rata-rata     : {total/success:.1f}s per gambar")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # python test_inpaint.py              → dummy test
    # python test_inpaint.py single image.png mask.png → single file
    # python test_inpaint.py batch        → batch folder

    if len(sys.argv) == 1:
        test_dummy()

    elif sys.argv[1] == "single" and len(sys.argv) == 4:
        test_single(sys.argv[2], sys.argv[3])

    elif sys.argv[1] == "batch":
        test_batch()

    else:
        print("Usage:")
        print("  python test_inpaint.py")
        print("  python test_inpaint.py single <image_path> <mask_path>")
        print("  python test_inpaint.py batch")