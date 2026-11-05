import pytest

from core.runtime_config import (
    MAX_CONCURRENT_FOLDER_JOBS,
    MAX_CONCURRENT_DOWNLOADS,
    MAX_DOWNLOAD_BYTES,
    MAX_SOURCE_PIXELS,
    RUNPOD_MAX_IN_FLIGHT,
    positive_int_env,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 10),
        ("16", 16),
        ("1", 1),
        ("0", 10),
        ("-2", 10),
        ("invalid", 10),
        ("", 10),
    ],
)
def test_positive_int_env_uses_safe_default(raw, expected):
    environ = {} if raw is None else {"WORKER_LIMIT": raw}
    assert positive_int_env("WORKER_LIMIT", 10, environ) == expected


def test_runtime_defaults_match_single_folder_ten_worker_topology():
    assert RUNPOD_MAX_IN_FLIGHT == 10
    assert MAX_CONCURRENT_FOLDER_JOBS == 1
    assert MAX_CONCURRENT_DOWNLOADS == 3
    assert MAX_DOWNLOAD_BYTES == 30 * 1024 * 1024
    assert MAX_SOURCE_PIXELS == 50_000_000


def test_positive_int_env_rejects_invalid_default():
    with pytest.raises(ValueError):
        positive_int_env("WORKER_LIMIT", 0, {})
