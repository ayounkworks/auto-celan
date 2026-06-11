# ============================================================
# handler.py
# RunPod Serverless + Simple LaMa
# ============================================================

import base64
import time
import traceback
from io import BytesIO

import runpod
from PIL import Image
from simple_lama_inpainting import SimpleLama


# ============================================================
# INIT
# ============================================================

print("Loading Simple LaMa...")

lama = SimpleLama()

print("Simple LaMa ready.")


# ============================================================
# BASE64
# ============================================================

def strip_prefix(data: str) -> str:

    if not data:
        raise ValueError("empty input")

    data = data.strip()

    if "," in data:
        data = data.split(",", 1)[1]

    # repair padding
    data += "=" * (-len(data) % 4)

    return data


def decode_rgb(data: str) -> Image.Image:

    raw = base64.b64decode(
        strip_prefix(data)
    )

    return Image.open(
        BytesIO(raw)
    ).convert("RGB")


def decode_mask(data: str) -> Image.Image:

    raw = base64.b64decode(
        strip_prefix(data)
    )

    return Image.open(
        BytesIO(raw)
    ).convert("L")


def encode_png(img: Image.Image) -> str:

    buf = BytesIO()

    img.save(
        buf,
        format="PNG",
    )

    return (
        "data:image/png;base64,"
        + base64.b64encode(
            buf.getvalue()
        ).decode("utf-8")
    )


# ============================================================
# HANDLER
# ============================================================

def handler(job):

    try:

        start_time = time.time()

        inp = job["input"]

        print("Decoding input...")

        image = decode_rgb(
            inp["image"]
        )

        mask = decode_mask(
            inp["mask"]
        )

        print(
            "Image:",
            image.size
        )

        print(
            "Mask:",
            mask.size
        )

        print("Running Simple LaMa...")

        result = lama(
            image,
            mask
        )

        elapsed = time.time() - start_time

        print(
            f"Done in {elapsed:.2f}s."
        )

        return {
            "image":
                encode_png(result)
        }

    except Exception as e:

        print(
            "Handler error:",
            repr(e)
        )

        traceback.print_exc()

        return {
            "error":
                str(e)
        }


# ============================================================
# START
# ============================================================

runpod.serverless.start(
    {
        "handler":
            handler
    }
)