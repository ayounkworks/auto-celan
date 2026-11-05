"""Small, dependency-free job cancellation and finalization rules."""

def cancellation_requested(job_id: str, cancelled_jobs: set[str]) -> bool:
    return job_id in cancelled_jobs


def determine_final_status(
    *,
    cancel_requested: bool,
    success_count: int,
    failed_count: int,
    pending_count: int,
) -> str:
    if cancel_requested:
        return "cancelled"
    if success_count == 0 and (failed_count > 0 or pending_count > 0):
        return "failed"
    if failed_count > 0:
        return "completed_with_errors"
    return "completed"


def output_is_available(status: str, output_folder_id: str | None) -> bool:
    return bool(output_folder_id) and status in (
        "completed",
        "completed_with_errors",
        "cancelled",
    )
