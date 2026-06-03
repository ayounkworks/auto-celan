# ============================================================
# PyTorch + CUDA
# ============================================================
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app


# ============================================================
# System deps
# ============================================================
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# Simple LaMa
# ============================================================
RUN pip install --no-cache-dir --no-deps \
    simple-lama-inpainting==0.1.2


# ============================================================
# Python deps
# ============================================================
COPY requirements.txt .

RUN pip install --no-cache-dir \
    -r requirements.txt


# ============================================================
# Download model
# ============================================================
COPY download_model.py .

RUN python download_model.py


# ============================================================
# Handler
# ============================================================
COPY handler.py .


ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]