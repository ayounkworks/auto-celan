FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download LaMa model
COPY download_model.py .
RUN python download_model.py

# Copy YOLO model
COPY best.pt /app/models/best.pt

# Handler
COPY handler.py .

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "handler.py"]
