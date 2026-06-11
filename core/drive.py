# ============================================================
# core/drive.py
# Google Drive: download, upload, folder helpers
# FIXED: thread-safe credential refresh dengan Lock
# ============================================================

import io
import os
import re
import time
import json
import threading
import asyncio
import aiohttp

from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from google.oauth2.credentials import Credentials

from core.config import DRIVE_OUTPUT_FOLDER_ID, VALID_EXTENSIONS

# ── Credentials ───────────────────────────────────────────

TOKEN_PATH = "token_drive.json"
CREDS      = None

# FIX: Lock untuk cegah concurrent credential refresh (race condition)
_creds_lock = threading.Lock()

print("[DEPLOY_CONFIRM] Checking credentials...")

env_token = os.getenv("DRIVE_TOKEN_JSON")
if env_token:
    print("Using DRIVE_TOKEN_JSON from Environment Variables.")
    info  = json.loads(env_token)
    CREDS = Credentials.from_authorized_user_info(info)
elif os.path.exists(TOKEN_PATH):
    print(f"Using {TOKEN_PATH} from local file.")
    with open(TOKEN_PATH, "r") as f:
        info  = json.load(f)
        CREDS = Credentials.from_authorized_user_info(info)
else:
    raise FileNotFoundError(
        "Credentials not found! Set DRIVE_TOKEN_JSON env var atau sediakan token_drive.json."
    )

_drive_local = threading.local()


def _refresh_creds_if_expired():
    """Thread-safe credential refresh — hanya satu thread refresh sekaligus."""
    if not (CREDS and CREDS.expired and CREDS.refresh_token):
        return
    with _creds_lock:
        # Double-check setelah dapat lock
        if CREDS.expired and CREDS.refresh_token:
            import google.auth.transport.requests
            CREDS.refresh(google.auth.transport.requests.Request())


def _new_drive():
    _refresh_creds_if_expired()
    return build("drive", "v3", credentials=CREDS)


def _get_drive():
    _refresh_creds_if_expired()
    if not hasattr(_drive_local, "client"):
        _drive_local.client = _new_drive()
    return _drive_local.client


# ── Helpers ───────────────────────────────────────────────

def extract_folder_id(link):
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', link)
    if match:
        return match.group(1)
    return link.split('/')[-1].split('?')[0]


def create_output_folder(parent_id, prefix="output"):
    if not parent_id:
        print("[WARNING] DRIVE_OUTPUT_FOLDER_ID tidak diset di .env, mencoba membuat di Root.")
    # FIX: format output_YYYYMMDD_HHMMSS (timestamp compact, no input folder name)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{ts}"
    meta = {
        "name"    : name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents" : [parent_id],
    }
    folder = _get_drive().files().create(body=meta, fields="id").execute()
    return name, folder.get("id")


def download_file(file_id):
    svc     = _get_drive()
    request = svc.files().get_media(fileId=file_id)
    buffer  = io.BytesIO()
    dl      = MediaIoBaseDownload(buffer, request)
    done    = False
    while not done:
        _, done = dl.next_chunk()
    buffer.seek(0)
    return buffer


def upload_file(file_bytes, filename, folder_id, mimetype="image/jpeg", _retries=3):
    last_err = None
    for attempt in range(_retries):
        try:
            svc = _new_drive()
            file_bytes.seek(0)
            if not folder_id:
                raise ValueError("Folder ID untuk upload tidak boleh None")
            metadata = {"name": filename, "parents": [folder_id]}
            media    = MediaIoBaseUpload(file_bytes, mimetype=mimetype, resumable=True)
            uploaded = svc.files().create(
                body=metadata, media_body=media, fields="id"
            ).execute(num_retries=2)
            return uploaded.get("id")
        except Exception as e:
            last_err = e
            print(f"[upload_file] attempt {attempt+1}/{_retries} failed: {e}")
            if attempt < _retries - 1:
                time.sleep(3 * (attempt + 1))
    raise last_err


def delete_folder(folder_id):
    try:
        _get_drive().files().delete(fileId=folder_id).execute()
        return True
    except Exception as e:
        print(f"Failed to delete folder {folder_id}: {e}")
        return False


def filter_and_sort_files(files):
    valid   = []
    skipped = []
    for f in files:
        ext = os.path.splitext(f["name"].lower())[1]
        if ext in VALID_EXTENSIONS:
            valid.append(f)
        else:
            skipped.append(f["name"])

    def natural_key(f):
        parts = re.split(r'(\d+)', f["name"])
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    valid.sort(key=natural_key)
    return valid, skipped


async def download_file_async(
    file_id: str,
    http_session: aiohttp.ClientSession,
    timeout: int = 60,
) -> io.BytesIO:
    """
    Download file dari Drive secara async.
    FIX: tambah timeout individual agar satu file hang tidak blokir batch.
    FIX: thread-safe refresh sebelum ambil token.
    """
    # Refresh di thread terpisah agar tidak blokir event loop
    await asyncio.to_thread(_refresh_creds_if_expired)

    url     = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    headers = {"Authorization": f"Bearer {CREDS.token}"}

    async with http_session.get(
        url,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Drive download failed: {resp.status} for {file_id}")
        data = await resp.read()
        return io.BytesIO(data)