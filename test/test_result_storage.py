import pytest

from runpod.result_storage import (
    ensure_base64_fallback_allowed,
    get_storage_config,
)


def test_storage_is_disabled_when_no_credentials_are_configured():
    assert get_storage_config({}) is None


def test_storage_config_requires_all_credentials():
    with pytest.raises(ValueError, match="Incomplete result storage"):
        get_storage_config({"RESULT_S3_BUCKET": "bucket"})


def test_storage_config_reads_r2_compatible_settings():
    config = get_storage_config(
        {
            "RESULT_S3_ENDPOINT_URL": "https://account.r2.cloudflarestorage.com",
            "RESULT_S3_BUCKET": "webtoon-results",
            "RESULT_S3_ACCESS_KEY_ID": "access",
            "RESULT_S3_SECRET_ACCESS_KEY": "secret",
            "RESULT_S3_REGION": "auto",
            "RESULT_URL_TTL_SECONDS": "1800",
        }
    )
    assert config["bucket"] == "webtoon-results"
    assert config["region"] == "auto"
    assert config["url_ttl_seconds"] == 1800


def test_invalid_presigned_url_ttl_uses_safe_default():
    config = get_storage_config(
        {
            "RESULT_S3_ENDPOINT_URL": "https://s3.example.com",
            "RESULT_S3_BUCKET": "bucket",
            "RESULT_S3_ACCESS_KEY_ID": "access",
            "RESULT_S3_SECRET_ACCESS_KEY": "secret",
            "RESULT_URL_TTL_SECONDS": "invalid",
        }
    )
    assert config["url_ttl_seconds"] == 3600


def test_base64_fallback_allows_small_output():
    ensure_base64_fallback_allowed(1024, 4 * 1024 * 1024)


def test_base64_fallback_rejects_large_output_with_setup_guidance():
    with pytest.raises(RuntimeError, match="Configure RESULT_S3_"):
        ensure_base64_fallback_allowed(5 * 1024 * 1024, 4 * 1024 * 1024)


@pytest.mark.parametrize(("payload", "limit"), [(-1, 10), (1, 0)])
def test_base64_fallback_rejects_invalid_limits(payload, limit):
    with pytest.raises(ValueError):
        ensure_base64_fallback_allowed(payload, limit)
