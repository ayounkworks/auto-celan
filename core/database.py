# ============================================================
# core/database.py
# FIXED: tambah db_increment_completed() — UPDATE atomic
#        tambah db_remove_pending_deletion() yang hilang dari import
# ============================================================

import sqlite3
import threading
import json
from datetime import datetime, timedelta

DB_PATH   = "manga_bot.db"
_db_local = threading.local()


def get_db():
    if not hasattr(_db_local, "conn"):
        _db_local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _db_local.conn.row_factory = sqlite3.Row
    return _db_local.conn


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-8000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            discord_id   TEXT PRIMARY KEY,
            username     TEXT,
            credit       INTEGER DEFAULT 0,
            total_used   INTEGER DEFAULT 0,
            is_banned    INTEGER DEFAULT 0,
            last_job_at  TEXT,
            joined_at    TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS credit_transactions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            discord_id   TEXT,
            amount       INTEGER,
            reason       TEXT,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS jobs (
            job_id           TEXT PRIMARY KEY,
            discord_id       TEXT,
            status           TEXT DEFAULT 'queued',
            folder_url       TEXT,
            output_folder_id TEXT,
            result_folder    TEXT,
            total_files      INTEGER DEFAULT 0,
            completed_files  INTEGER DEFAULT 0,
            failed_files     TEXT DEFAULT '[]',
            credit_used      INTEGER DEFAULT 0,
            credit_held      INTEGER DEFAULT 0,
            started_at       TEXT DEFAULT CURRENT_TIMESTAMP,
            finished_at      TEXT,
            log              TEXT DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS processed_files (
            job_id    TEXT,
            filename  TEXT,
            status    TEXT,
            duration  REAL,
            PRIMARY KEY (job_id, filename)
        );
        CREATE TABLE IF NOT EXISTS pending_deletions (
            folder_id TEXT PRIMARY KEY,
            delete_at TEXT
        );
    """)
    conn.commit()
    conn.close()


# ── User / Credit ─────────────────────────────────────────

def db_get_user(discord_id):
    return get_db().execute(
        "SELECT * FROM users WHERE discord_id = ?", (discord_id,)
    ).fetchone()


def db_register_user(discord_id, username):
    existing = db_get_user(discord_id)
    if existing:
        return existing
    conn = get_db()
    conn.execute(
        "INSERT INTO users (discord_id, username, credit) VALUES (?, ?, ?)",
        (discord_id, username, 0)
    )
    conn.execute(
        "INSERT INTO credit_transactions (discord_id, amount, reason) VALUES (?, ?, ?)",
        (discord_id, 0, "initial_user")
    )
    conn.commit()
    return db_get_user(discord_id)


def db_get_credit(discord_id):
    user = db_get_user(discord_id)
    return user["credit"] if user else 0


def db_add_credit(discord_id, amount, reason):
    conn = get_db()
    conn.execute(
        "UPDATE users SET credit = credit + ? WHERE discord_id = ?",
        (amount, discord_id)
    )
    conn.execute(
        "INSERT INTO credit_transactions (discord_id, amount, reason) VALUES (?, ?, ?)",
        (discord_id, amount, reason)
    )
    conn.commit()


def db_hold_credit(discord_id, amount):
    conn = get_db()
    conn.execute(
        "UPDATE users SET credit = credit - ? WHERE discord_id = ?",
        (amount, discord_id)
    )
    conn.commit()


def db_get_history(discord_id, limit=10):
    return get_db().execute(
        """SELECT amount, reason, created_at
           FROM credit_transactions
           WHERE discord_id = ?
           ORDER BY created_at DESC LIMIT ?""",
        (discord_id, limit)
    ).fetchall()


def db_is_banned(discord_id):
    user = db_get_user(discord_id)
    return bool(user["is_banned"]) if user else False


def db_set_banned(discord_id, banned):
    conn = get_db()
    conn.execute(
        "UPDATE users SET is_banned = ? WHERE discord_id = ?",
        (1 if banned else 0, discord_id)
    )
    conn.commit()


def db_get_cooldown_remaining(discord_id):
    return 0


def db_update_last_job(discord_id):
    conn = get_db()
    conn.execute(
        "UPDATE users SET last_job_at = ? WHERE discord_id = ?",
        (datetime.now().isoformat(), discord_id)
    )
    conn.commit()


# ── Job Tracking ──────────────────────────────────────────

def db_create_job(job_id, discord_id, folder_url, credit_held):
    conn = get_db()
    conn.execute(
        """INSERT INTO jobs (job_id, discord_id, folder_url, credit_held, status)
           VALUES (?, ?, ?, ?, 'queued')""",
        (job_id, discord_id, folder_url, credit_held)
    )
    conn.commit()


def db_update_job(job_id, **kwargs):
    if not kwargs:
        return
    conn = get_db()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [job_id]
    conn.execute(f"UPDATE jobs SET {sets} WHERE job_id = ?", vals)
    conn.commit()


def db_increment_completed(job_id) -> int:
    """Atomic increment — cegah race condition di concurrent tasks."""
    conn = get_db()
    conn.execute(
        "UPDATE jobs SET completed_files = completed_files + 1 WHERE job_id = ?",
        (job_id,)
    )
    conn.commit()
    row = conn.execute(
        "SELECT completed_files FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    return row["completed_files"] if row else 0


def db_get_job(job_id):
    return get_db().execute(
        "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()


def db_append_log(job_id, message):
    conn = get_db()
    row  = conn.execute(
        "SELECT log FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row:
        log = json.loads(row["log"] or "[]")
        log.append(message)
        conn.execute(
            "UPDATE jobs SET log = ? WHERE job_id = ?",
            (json.dumps(log[-50:]), job_id)
        )
        conn.commit()


def db_mark_file_processed(job_id, filename, status, duration):
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO processed_files "
        "(job_id, filename, status, duration) VALUES (?, ?, ?, ?)",
        (job_id, filename, status, duration)
    )
    conn.commit()


def db_get_processed_files(job_id):
    rows = get_db().execute(
        "SELECT filename FROM processed_files WHERE job_id = ?", (job_id,)
    ).fetchall()
    return {r["filename"] for r in rows}


def db_schedule_deletion(folder_id):
    """
    Jadwalkan penghapusan folder setelah 15 menit.
    deletion_loop() di main bot event loop yang akan eksekusi.
    """
    delete_at = (datetime.now() + timedelta(minutes=15)).isoformat()
    conn      = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO pending_deletions (folder_id, delete_at) VALUES (?, ?)",
        (folder_id, delete_at)
    )
    conn.commit()


def db_get_pending_deletions():
    """Ambil folder yang sudah waktunya dihapus."""
    rows = get_db().execute(
        "SELECT folder_id FROM pending_deletions WHERE delete_at <= ?",
        (datetime.now().isoformat(),)
    ).fetchall()
    return [r["folder_id"] for r in rows]


def db_remove_pending_deletion(folder_id):
    conn = get_db()
    conn.execute(
        "DELETE FROM pending_deletions WHERE folder_id = ?", (folder_id,)
    )
    conn.commit()