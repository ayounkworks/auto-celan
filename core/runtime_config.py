"""Shared runtime limits for the Discord and FastAPI entry points.

These limits are process-local. Deploy only one application entry point per
Railway service unless a distributed queue/lock is introduced.
"""
import os
from typing import Mapping, Optional

def positive_int_env(
    name: str,
    default: int,
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    """Read a positive integer environment variable; fall back safely."""
    if default < 1:
        raise ValueError("default must be a positive integer")
    source = os.environ if environ is None else environ
    raw = source.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default

# Match the intended deployment topology: one folder job can fan out to
# the configured RunPod worker pool.
RUNPOD_MAX_IN_FLIGHT = positive_int_env("RUNPOD_MAX_IN_FLIGHT", 10)
MAX_CONCURRENT_FOLDER_JOBS = positive_int_env("MAX_CONCURRENT_FOLDER_JOBS", 1)

# Bound concurrent Drive downloads and per-image memory use.
MAX_CONCURRENT_DOWNLOADS = positive_int_env("MAX_CONCURRENT_DOWNLOADS", 3)
MAX_DOWNLOAD_BYTES = positive_int_env("MAX_DOWNLOAD_BYTES", 30 * 1024 * 1024)
MAX_SOURCE_PIXELS = positive_int_env("MAX_SOURCE_PIXELS", 50_000_000)
