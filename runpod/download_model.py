import shutil
import os
from simple_lama_inpainting.models.model import download_model, LAMA_MODEL_URL

os.makedirs("/app/models", exist_ok=True)
path = download_model(LAMA_MODEL_URL)
shutil.copy(path, "/app/models/big-lama.pt")
size = os.path.getsize("/app/models/big-lama.pt")
print(f"Model baked into image: {size} bytes ({size/1024/1024:.1f} MB)")
