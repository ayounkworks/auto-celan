# ============================================================
# main.py  ← Entry point utama
# YOLO edition — Vision API dihapus
#
# Cara run:
#   python main.py local
#   python main.py
# ============================================================

import sys
import os

# ── Mode Lokal ────────────────────────────────────────────
if len(sys.argv) > 1 and sys.argv[1] == "local":
    from local_runner import main
    main()
    sys.exit(0)

# ── Mode Server ───────────────────────────────────────────
import asyncio
import gc
import io
import json
import uuid
import socket
from contextlib import asynccontextmanager
from datetime import datetime

import aiohttp
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from core.config import DRIVE_OUTPUT_FOLDER_ID
from core.database import init_db, db_create_job, db_get_job
from core.drive import extract_folder_id, filter_and_sort_files, upload_file, _get_drive
from core.image_processing import progress_bar, format_eta

import core.runpod_client as runpod_module
import core.pipeline as pipeline_module
from core.pipeline import jobs, job_queue, cancelled_jobs, run_pipeline, deletion_loop, warmup


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    runpod_module.runpod_sem     = asyncio.Semaphore(8)
    pipeline_module._warmup_lock = asyncio.Lock()
    pipeline_module.pipeline_sem = asyncio.Semaphore(10)

    timeout   = aiohttp.ClientTimeout(total=300, connect=20, sock_connect=20, sock_read=300)
    connector = aiohttp.TCPConnector(limit=30, ttl_dns_cache=300, family=socket.AF_INET)
    pipeline_module._http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    del_task = asyncio.create_task(deletion_loop())
    asyncio.create_task(warmup())

    yield

    await pipeline_module._http_session.close()
    del_task.cancel()


app = FastAPI(title="Manga Text Removal API", lifespan=lifespan)


class CleanRequest(BaseModel):
    folder_url: str


class ScanRequest(BaseModel):
    folder_url: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scan")
async def scan_folder_endpoint(req: ScanRequest):
    try:
        folder_id = extract_folder_id(req.folder_url)

        def _list():
            return _get_drive().files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="files(id, name)"
            ).execute()

        result = await asyncio.to_thread(_list)
        valid, skipped = filter_and_sort_files(result.get("files", []))
        return {
            "file_count":   len(valid),
            "skipped_count": len(skipped),
            "filenames":    [f["name"] for f in valid],
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/upload_images")
async def api_upload_images_form(
    label: str = Form(default=""),
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(400, "Tidak ada file")

    ts          = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = label or f"Upload_{ts}"
    meta = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if DRIVE_OUTPUT_FOLDER_ID:
        meta["parents"] = [DRIVE_OUTPUT_FOLDER_ID]
    folder    = await asyncio.to_thread(
        lambda: _get_drive().files().create(body=meta, fields="id").execute()
    )
    folder_id = folder.get("id")
    uploaded  = []

    for f in files:
        data    = await f.read()
        buf     = io.BytesIO(data)
        ext     = os.path.splitext(f.filename.lower())[1]
        mime    = ("image/png" if ext == ".png"
                   else "image/webp" if ext == ".webp" else "image/jpeg")
        await asyncio.to_thread(upload_file, buf, f.filename, folder_id, mime)
        uploaded.append(f.filename)

    return {
        "folder_url": f"https://drive.google.com/drive/folders/{folder_id}",
        "folder_id":  folder_id,
        "uploaded":   len(uploaded),
        "filenames":  uploaded,
    }


@app.post("/clean")
async def api_clean(req: CleanRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    db_create_job(job_id, "", req.folder_url, 0)
    jobs[job_id] = {
        "status": "queued", "progress": "Starting...",
        "result_folder": None, "output_folder_id": None,
        "queue_position": len(job_queue) + 1,
        "total_files": 0, "completed_files": 0,
        "failed_files": [], "eta": "menghitung...", "log": [],
    }
    bg.add_task(run_pipeline, job_id, req.folder_url)
    return {"job_id": job_id, "message": "Pipeline started"}


@app.get("/job/{job_id}")
async def api_get_status(job_id: str):
    data = jobs.get(job_id)
    if not data:
        row = db_get_job(job_id)
        if not row:
            raise HTTPException(404, "Job tidak ditemukan")
        data = dict(row)
    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
