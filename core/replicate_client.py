# ============================================================
# core/replicate_client.py
# Replicate LaMa client
# ============================================================

import replicate
import io
import time
import asyncio
import aiohttp
from PIL import Image
from core.config import REPLICATE_API_TOKEN, REPLICATE_MODEL

async def run_replicate_lama(image: Image.Image, mask: Image.Image, label: str = "", http_session: aiohttp.ClientSession = None) -> Image.Image:
    if not REPLICATE_API_TOKEN:
        print(f"[{label}] Error: REPLICATE_API_TOKEN tidak ditemukan")
        return None

    start_time = time.time()
    try:
        # Convert PIL to bytes
        img_buf = io.BytesIO()
        image.save(img_buf, format="PNG")
        img_buf.seek(0)
        
        mask_buf = io.BytesIO()
        mask.save(mask_buf, format="PNG")
        mask_buf.seek(0)

        # Run via thread pool
        def _call():
            client = replicate.Client(api_token=REPLICATE_API_TOKEN)
            return client.run(
                REPLICATE_MODEL,
                input={"image": img_buf, "mask": mask_buf}
            )

        output_url = await asyncio.to_thread(_call)
        if not output_url: return None

        # Download result
        if http_session:
            async with http_session.get(output_url) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    print(f"[{label}] Replicate finished in {time.time() - start_time:.2f}s")
                    return Image.open(io.BytesIO(data)).convert("RGB")
        
    except Exception as e:
        print(f"[{label}] Replicate error: {e}")
    
    return None
