FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080

# RUN_MODE=bot  → python bot.py
# RUN_MODE=api  → uvicorn main:app (default)
CMD ["sh", "-c", \
  "if [ \"$RUN_MODE\" = 'bot' ]; then python bot.py; \
   else uvicorn main:app --host 0.0.0.0 --port ${PORT}; fi"]
