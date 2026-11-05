"""Pure retry policy for RunPod submission and status polling."""


RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})


def is_retryable_http_status(status: int) -> bool:
    return status in RETRYABLE_HTTP_STATUSES


def poll_backoff_seconds(error_count: int, base_seconds: float = 1.0, max_seconds: float = 8.0) -> float:
    if error_count < 1:
        raise ValueError("error_count must be at least 1")
    if base_seconds <= 0 or max_seconds <= 0:
        raise ValueError("backoff values must be positive")
    return min(base_seconds * (2 ** (error_count - 1)), max_seconds)
