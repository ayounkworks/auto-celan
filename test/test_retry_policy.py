import pytest

from core.retry_policy import is_retryable_http_status, poll_backoff_seconds


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_http_statuses_are_retryable(status):
    assert is_retryable_http_status(status)


@pytest.mark.parametrize("status", [200, 400, 401, 403, 404, 422])
def test_non_transient_http_statuses_are_not_retryable(status):
    assert not is_retryable_http_status(status)


def test_poll_backoff_grows_exponentially_and_is_capped():
    assert poll_backoff_seconds(1) == 1.0
    assert poll_backoff_seconds(2) == 2.0
    assert poll_backoff_seconds(3) == 4.0
    assert poll_backoff_seconds(4) == 8.0
    assert poll_backoff_seconds(8) == 8.0


@pytest.mark.parametrize(
    ("count", "base", "maximum"),
    [(0, 1, 8), (1, 0, 8), (1, 1, 0)],
)
def test_invalid_backoff_inputs_are_rejected(count, base, maximum):
    with pytest.raises(ValueError):
        poll_backoff_seconds(count, base, maximum)
