# Auto Celan — Local CLI

Auto Celan processes Webtoon page images from a local folder and writes cleaned images to a local output folder. There is no Railway service, Discord bot, FastAPI server, or Google Drive dependency in this CLI version.

**GPU processing still runs on RunPod Serverless.** The CLI and file management run on your PC; each image is sent to the configured RunPod endpoint for YOLO detection and LaMa inpainting. This keeps the laptop from needing a capable NVIDIA GPU.

## Requirements

- Python 3.10+
- A RunPod Serverless endpoint containing the compatible YOLO + LaMa handler
- RunPod API key and endpoint ID

## Setup (Windows PowerShell)

```powershell
git clone https://github.com/ayounkworks/auto-celan.git
cd auto-celan
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`:

```env
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id
```

Keep `.env` private. Never commit API keys.

## Run

Pass the folder path directly:

```powershell
python main.py "D:\Webtoon\Chapter 01"
```

Default output is a sibling folder named `Chapter 01_cleaned`.

Optional output path and concurrency:

```powershell
python main.py "D:\Webtoon\Chapter 01" --output "D:\Webtoon\Chapter 01_result" --workers 2
```

Or use the default `input` folder:

```powershell
python main.py
```

Supported extensions: JPG, JPEG, PNG, WEBP. Files are processed in natural filename order (page1, page2, page10). Nested subdirectories are not scanned. Failed images are reported and do not prevent the remaining files from running.

## Important behavior

- Source images remain untouched.
- Output is written locally and the output folder is created automatically.
- The CLI does not require Google Drive credentials.
- `--workers` controls the number of in-flight RunPod jobs. Start with 1–2; increase only after checking endpoint capacity and costs.
- RunPod usage is billed separately by your RunPod plan/endpoint configuration.
- The local machine still needs a stable internet connection to send images and download processed results.

## Troubleshooting

- `RUNPOD_API_KEY belum diatur`: add the key to `.env` in the project root.
- HTTP 401: verify the RunPod API key.
- HTTP 404: verify `RUNPOD_ENDPOINT_ID`.
- Worker/GPU unavailable: check the RunPod endpoint status and worker logs.
- A file reported as failed is not silently replaced with the uncleaned original; inspect the error and retry it.

## Run tests

```powershell
python -m pip install pytest
python -m compileall -q .
python -m pytest -q
```

## Project components

- `main.py`: CLI entry point.
- `local_runner.py`: local file orchestration, bounded concurrency, output encoding.
- `core/local_paths.py`: image discovery and default output path.
- `core/runpod_client.py`: RunPod Serverless HTTP client.
- `runpod/`: worker implementation and model image used by the RunPod endpoint.

Legacy Railway/Discord/Google Drive source modules and deployment files are not part of this CLI branch.
