# ============================================================
# download_model.py
# Download big-lama.pt dari GitHub releases
# Tidak butuh dependency apapun selain stdlib
# ============================================================

import os
import urllib.request

MODEL_DIR  = "/app/models"
MODEL_PATH = os.path.join(MODEL_DIR, "big-lama.pt")
MODEL_URL  = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Downloading big-lama.pt...")
urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
print(f"✅ Saved to {MODEL_PATH} ({size_mb:.1f} MB)")