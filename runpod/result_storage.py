"""S3-compatible result storage for RunPod output payloads.

Supports AWS S3 and S3-compatible providers such as Cloudflare R2.
The SDK is imported lazily so pure policy tests do not require boto3.
"""
import os
import re
import uuid
from typing import Mapping, Optional


_REQUIRED_ENV = (
    "RESULT_S3_ENDPOINT_URL",
    "RESULT_S3_BUCKET",
    "RESULT_S3_ACCESS_KEY_ID",
    "RESULT_S3_SECRET_ACCESS_KEY",
)


def get_storage_config(environ: Optional[Mapping[str, str]] = None):
    source = os.environ if environ is None else environ
    values = {key: source.get(key, "").strip() for key in _REQUIRED_ENV}
    configured = [bool(value) for value in values.values()]
    if not any(configured):
        return None
    if not all(configured):
        missing = [key for key, value in values.items() if not value]
        raise ValueError(
            "Incomplete result storage configuration; missing: " + ", ".join(missing)
        )
    return {
        "endpoint_url": values["RESULT_S3_ENDPOINT_URL"],
        "bucket": values["RESULT_S3_BUCKET"],
        "access_key_id": values["RESULT_S3_ACCESS_KEY_ID"],
        "secret_access_key": values["RESULT_S3_SECRET_ACCESS_KEY"],
        "region": source.get("RESULT_S3_REGION", "auto").strip() or "auto",
        "url_ttl_seconds": _positive_int(
            source.get("RESULT_URL_TTL_SECONDS", "3600"), 3600
        ),
    }


def _positive_int(raw: str, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def ensure_base64_fallback_allowed(payload_bytes: int, max_bytes: int) -> None:
    if payload_bytes < 0 or max_bytes < 1:
        raise ValueError("payload_bytes must be non-negative and max_bytes positive")
    if payload_bytes > max_bytes:
        raise RuntimeError(
            f"Result is {payload_bytes:,} bytes, above the Base64 fallback limit "
            f"of {max_bytes:,} bytes. Configure RESULT_S3_* object storage."
        )


def upload_result_bytes(
    payload: bytes,
    job_id: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
):
    config = get_storage_config(environ)
    if config is None:
        return None

    import boto3

    safe_job_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(job_id or "unknown"))[:80]
    key = f"auto-celan/{safe_job_id}/{uuid.uuid4().hex}.jpg"
    client = boto3.client(
        "s3",
        endpoint_url=config["endpoint_url"],
        aws_access_key_id=config["access_key_id"],
        aws_secret_access_key=config["secret_access_key"],
        region_name=config["region"],
    )
    client.put_object(
        Bucket=config["bucket"],
        Key=key,
        Body=payload,
        ContentType="image/jpeg",
    )
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": config["bucket"], "Key": key},
        ExpiresIn=config["url_ttl_seconds"],
    )
    return {"url": url, "key": key, "bytes": len(payload)}
