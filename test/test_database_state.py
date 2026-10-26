import json
from datetime import datetime

import pytest

import core.database as database


@pytest.fixture
def isolated_database(tmp_path, monkeypatch):
    # Each test receives a separate SQLite database and connection.
    connection = getattr(database._db_local, "conn", None)
    if connection is not None:
        connection.close()
        delattr(database._db_local, "conn")

    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "manga_bot.db"))
    database.init_db()
    yield database

    connection = getattr(database._db_local, "conn", None)
    if connection is not None:
        connection.close()
        delattr(database._db_local, "conn")


def test_resume_skips_only_successful_files(isolated_database):
    db = isolated_database
    db.db_register_user("user-1", "tester")
    db.db_create_job("job-1", "user-1", "drive-folder", 0)

    db.db_mark_file_processed("job-1", "chapter-01.png", "failed", 1.0)
    assert db.db_get_processed_files("job-1") == set()

    db.db_mark_file_processed("job-1", "chapter-01.png", "success", 0.8)
    assert db.db_get_processed_files("job-1") == {"chapter-01.png"}


def test_processed_count_is_unique_and_failure_list_can_clear(isolated_database):
    db = isolated_database
    db.db_register_user("user-1", "tester")
    db.db_create_job("job-1", "user-1", "drive-folder", 0)

    assert db.db_mark_file_processed("job-1", "chapter-01.png", "failed", 1.0) == 1
    db.db_record_file_failure("job-1", "chapter-01.png")
    db.db_record_file_failure("job-1", "chapter-01.png")

    assert db.db_mark_file_processed("job-1", "chapter-01.png", "success", 0.8) == 1
    db.db_clear_file_failure("job-1", "chapter-01.png")

    row = db.db_get_job("job-1")
    assert row["completed_files"] == 1
    assert json.loads(row["failed_files"]) == []


def test_default_deletion_schedule_is_24_hours(isolated_database):
    db = isolated_database
    before = datetime.now()
    db.db_schedule_deletion("output-folder")
    row = db.get_db().execute(
        "SELECT delete_at FROM pending_deletions WHERE folder_id = ?",
        ("output-folder",),
    ).fetchone()
    after = datetime.now()

    delete_at = datetime.fromisoformat(row["delete_at"])
    minutes_from_before = (delete_at - before).total_seconds() / 60
    minutes_from_after = (delete_at - after).total_seconds() / 60
    assert 1439 <= minutes_from_before <= 1441
    assert 1439 <= minutes_from_after <= 1441
