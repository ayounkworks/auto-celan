FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download big-lama model
RUN mkdir -p /app/models && \
    wget -q "https://huggingface.co/fashn-ai/LaMa/resolve/main/big-lama.pt" \
    -O /app/models/big-lama.pt

# Copy YOLO model
COPY best.pt /app/models/best.pt

# Handler
COPY handler.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
