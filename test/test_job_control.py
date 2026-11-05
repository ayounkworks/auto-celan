import pytest

from core.job_control import (
    cancellation_requested,
    determine_final_status,
    output_is_available,
)


@pytest.mark.parametrize(
    ("cancelled", "success", "failed", "pending", "expected"),
    [
        (True, 3, 1, 6, "cancelled"),
        (False, 0, 0, 5, "failed"),
        (False, 0, 2, 0, "failed"),
        (False, 3, 2, 0, "completed_with_errors"),
        (False, 5, 0, 0, "completed"),
        (False, 0, 0, 0, "completed"),
    ],
)
def test_final_status_accounts_for_cancellation_and_pending_files(
    cancelled, success, failed, pending, expected
):
    assert determine_final_status(
        cancel_requested=cancelled,
        success_count=success,
        failed_count=failed,
        pending_count=pending,
    ) == expected


def test_cancellation_flag_is_checked_by_job_id():
    assert cancellation_requested("job-1", {"job-1"})
    assert not cancellation_requested("job-2", {"job-1"})


@pytest.mark.parametrize("status", ["completed", "completed_with_errors", "cancelled"])
def test_partial_or_complete_output_is_available_for_terminal_jobs(status):
    assert output_is_available(status, "drive-folder-id")


@pytest.mark.parametrize("status", ["queued", "running", "failed"])
def test_output_is_not_advertised_for_non_result_statuses(status):
    assert not output_is_available(status, "drive-folder-id")


def test_output_is_not_advertised_without_folder_id():
    assert not output_is_available("cancelled", None)
