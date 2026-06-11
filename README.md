# auto_celan_v2 — Manga Text Removal

## Struktur Folder

```
auto_celan_v2/
│
├── main.py                  ← Entry point (server FastAPI + mode lokal)
├── local_runner.py          ← Logic mode lokal (python main.py local)
├── requirements.txt
├── .env                     ← API keys (JANGAN di-commit ke git)
├── token_drive.json         ← OAuth token Google Drive
├── manga_bot.db             ← SQLite database
│
├── core/                    ← Modul inti (mudah debug per-file)
│   ├── config.py            ← Semua konstanta & env vars
│   ├── database.py          ← SQLite helpers
│   ├── drive.py             ← Google Drive download/upload
│   ├── image_processing.py  ← smart_clean, SFX filter, crop, validate
│   ├── pipeline.py          ← Orchestrasi Vision → RunPod → Upload
│   └── runpod_client.py     ← HTTP ke RunPod Serverless API
│
├── runpod/                  ← Docker image untuk RunPod endpoint
│   ├── Dockerfile
│   ├── handler.py
│   ├── download_model.py
│   └── requirements.txt
│
├── input/                   ← Taruh gambar di sini untuk mode lokal
├── output/                  ← Hasil pembersihan mode lokal
└── test/                    ← Script testing
    └── test_inpaint.py
```

---

## Bug yang Diperbaiki

### [BUG-1] `RUNPOD_API_KEY` tidak pernah di-load dari `.env`
**File:** `core/config.py`
**Masalah:** Di `main.py` lama, variabel `RUNPOD_API_KEY` tidak ada di blok `os.getenv(...)`.
Akibatnya `run_runpod_lama()` selalu kirim Authorization header kosong → HTTP 401 dari RunPod.
**Fix:** Ditambahkan `RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")` di `core/config.py`.

### [BUG-2] `BytesIO` NameError di `run_runpod_lama()`
**File:** `core/runpod_client.py`
**Masalah:** Fungsi `run_runpod_lama()` pakai `BytesIO(...)` tapi yang di-import cuma `import io`.
Maka `BytesIO` adalah `NameError` saat runtime — RunPod tidak pernah bisa dipanggil.
**Fix:** `from io import BytesIO` ditambahkan eksplisit di `runpod_client.py`.

### [BUG-3] Double pemanggilan `create_output_folder` di pipeline
**File:** `core/pipeline.py`
**Masalah:** Copy-paste error → output folder dibuat dua kali, folder pertama tidak terpakai (leaked di Drive).
**Fix:** Satu call saja, import `DRIVE_OUTPUT_FOLDER_ID` di tempat yang tepat.

### [BUG-4] `_http_session` tidak pernah dibuat di mode lokal
**File:** `local_runner.py`
**Masalah:** Mode lokal langsung call `run_runpod_lama()` tanpa `_http_session`, padahal fungsi lama
mengharapkan session global dari lifespan. Kalau session None → crash atau buat session baru tiap file.
**Fix:** `local_runner.py` buat satu `aiohttp.ClientSession` dan inject ke semua task dalam satu `asyncio.run()`.

### [BUG-5] `asyncio.run(_http_session.close())` dipanggil setelah `asyncio.run()` sudah tutup
**File:** `main.py` lama (blok `__main__`)
**Masalah:** Setelah `asyncio.run(run_local_batch())` selesai, event loop sudah tutup.
Memanggil `asyncio.run(_http_session.close())` lagi → `RuntimeError: This event loop is already running`.
**Fix:** Di `local_runner.py` session dipakai dengan `async with` → otomatis tutup saat selesai.

### [BUG-6] `vision_client` tidak di-pass ke mode lokal dengan benar
**File:** `local_runner.py`
**Masalah:** Mode lokal lama mencoba buat `vision.ImageAnnotatorClient()` baru tanpa API key,
sehingga muncul `DefaultCredentialsError`.
**Fix:** `local_runner.py` inisialisasi `vision_client` sendiri dengan `client_options={"api_key": GOOGLE_API_KEY}`.

---

## Cara Run

### Mode Lokal (PowerShell / Terminal)
```powershell
# 1. Taruh gambar ke folder input/
# 2. Jalankan:
python main.py local
# 3. Hasil ada di folder output/
```

### Mode Server (Railway / Docker)
```bash
python main.py
# atau:
uvicorn main:app --host 0.0.0.0 --port 8080
```

---

## Apakah perlu push Docker lagi?

**Tidak perlu**, selama kamu hanya mengubah kode di sisi klien (main.py dan modul-modulnya).

Docker image di RunPod **hanya berisi**:
- `handler.py` — menerima request dari RunPod
- Model LaMa (`big-lama.pt`) — sudah di-bake saat `docker build`
- Dependencies Python untuk inference

Kamu perlu **rebuild & push Docker** hanya kalau:
| Perubahan | Perlu push Docker? |
|-----------|-------------------|
| Edit `main.py`, `core/*.py`, `local_runner.py` | ❌ Tidak |
| Edit `runpod/handler.py` | ✅ Ya |
| Ganti model LaMa | ✅ Ya |
| Tambah/ubah library di `runpod/requirements.txt` | ✅ Ya |

### Cara Push Update ke RunPod (Versioning)
1. **Build:** 
   `docker build -t ayounkwork/lama-handler:v7 ./runpod`
2. **Push:** 
   `docker push ayounkwork/lama-handler:v7`
3. **Deploy:** 
   - Buka RunPod Console
   - Pilih Endpoint -> Settings
   - Ubah `Container Image` ke tag baru (misal `:v7`)
   - Save & Update.

---

## Environment Variables (.env)

```env
GOOGLE_VISION_API_KEY=...
DRIVE_OUTPUT_FOLDER_ID=...
RUNPOD_API_KEY=rpa_...          ← WAJIB ada, ini yang bug di versi lama
RUNPOD_ENDPOINT_ID=3wjmfk65...  ← Opsional, ada default di config.py
```
