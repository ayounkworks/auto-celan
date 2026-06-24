# ============================================================
# download_model.py
# Download LaMa big-lama checkpoint dari advimman
# dan config untuk mode dengan refinement
# ============================================================

import os
import shutil
import urllib.request

MODEL_DIR      = "/app/models"
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "lama")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Download checkpoint ───────────────────────────────────
# big-lama adalah model terbesar dari advimman, support refinement
LAMA_CHECKPOINT_URL = (
    "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
)

print("Downloading big-lama checkpoint...")
zip_path = "/tmp/big-lama.zip"
urllib.request.urlretrieve(LAMA_CHECKPOINT_URL, zip_path)
print("Download complete.")

# Extract
import zipfile
print("Extracting...")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall("/tmp/lama_extracted")

# Cari best.ckpt dan models.yaml
extracted_dir = "/tmp/lama_extracted/big-lama"
for root, dirs, files in os.walk(extracted_dir):
    for f in files:
        src = os.path.join(root, f)
        dst = os.path.join(CHECKPOINT_DIR, f)
        shutil.copy(src, dst)
        print(f"Copied {f} → {CHECKPOINT_DIR}/")

os.remove(zip_path)
shutil.rmtree("/tmp/lama_extracted", ignore_errors=True)

# ── Tulis config.yaml untuk inference dengan refinement ───
CONFIG_PATH = os.path.join(MODEL_DIR, "config.yaml")

# Config ini dibuat dari config lama yang ada di repo advimman
# (configs/prediction/default.yaml + configs/training/big-lama-fourier.yaml)
config_yaml = """
# LaMa inference config — big-lama with refinement
# Digunakan oleh handler.py

training_model:
  kind: default
  num_iterations: 1000000
  rescale_schedule: false
  out_size: 512
  predict_only: true
  visualize_each_iters: 100
  concat_mask: true

visualizer:
  kind: noop
  outdir: /tmp/lama_vis
  enabled: false

generator:
  kind: ffc_resnet
  input_nc: 4
  output_nc: 3
  ngf: 64
  n_downsampling: 3
  n_blocks: 18
  add_out_act: sigmoid
  max_features: 1024
  latent_noise_upsample: false
  fourier_context_mode: false
  ffc_conv_combine: false

discriminator:
  kind: r3_multiscale
  input_nc: 3
  ndf: 64
  n_layers: 4
  norm_layer: none
  sigmoid_output: false

adversarial:
  kind: r3gan

losses:
  l1:
    weight_missing: 0
    weight_known: 10
  perceptual:
    weight: 0
    normalize_inputs: false
  resnet_pl:
    weight: 30
    weights_path: ${MODEL_DIR}/ade20k-resnet50dilated-ppm_deepsup.pth
    arch_encoder: resnet50dilated
    arch_decoder: ppm_deepsup
    segmentation: true
  feature_matching:
    weight: 100
  adversarial:
    weight: 10

optimizers:
  generator:
    kind: adam
    lr: 0.001
  discriminator:
    kind: adam
    lr: 0.0001

data:
  kind: default
  num_workers: 3
  batch_size: 10
  val_online_augmented: false

locations:
  baseDir: ${MODEL_DIR}
""".strip()

with open(CONFIG_PATH, "w") as f:
    f.write(config_yaml)
print(f"Config written to {CONFIG_PATH}")

# Verifikasi
ckpt = os.path.join(CHECKPOINT_DIR, "best.ckpt")
if os.path.exists(ckpt):
    size_mb = os.path.getsize(ckpt) / 1024 / 1024
    print(f"✅ Checkpoint: {ckpt} ({size_mb:.1f} MB)")
else:
    files = os.listdir(CHECKPOINT_DIR)
    print(f"WARNING: best.ckpt tidak ditemukan. Files di dir: {files}")
    # Coba rename file yang ada
    for f in files:
        if f.endswith(".ckpt") or f.endswith(".pt"):
            src = os.path.join(CHECKPOINT_DIR, f)
            dst = os.path.join(CHECKPOINT_DIR, "best.ckpt")
            shutil.copy(src, dst)
            print(f"Renamed {f} → best.ckpt")
            break

print("✅ Model setup complete.")
