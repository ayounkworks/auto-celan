import re

# ── pipeline.py ──────────────────────────────────────────
content = open('core/pipeline.py', 'r', encoding='utf-8').read()

content = content.replace(
    'db_increment_completed(job_id)\n                    if job_id in jobs:\n                        jobs[job_id]["completed_files"] = db_get_job(job_id)["completed_files"] or 0',
    'completed = db_increment_completed(job_id)\n                    if job_id in jobs:\n                        jobs[job_id]["completed_files"] = completed'
)

content = content.replace(
    '                if success:\n                    print(f"Auto-deleted: {folder_id}")\n                db_remove_pending_deletion(folder_id)',
    '                if success:\n                    db_remove_pending_deletion(folder_id)\n                    print(f"Auto-deleted: {folder_id}")\n                else:\n                    print(f"Delete failed, will retry: {folder_id}")'
)

open('core/pipeline.py', 'w', encoding='utf-8').write(content)
print('pipeline.py patched')

# ── Verifikasi semua file ────────────────────────────────
checks = [
    ('core/pipeline.py',       'completed = db_increment_completed'),
    ('core/pipeline.py',       'Delete failed, will retry'),
    ('core/art_protection.py', 'score += 2'),
    ('core/image_processing.py', 'min(15, max(3, h // 8'),
]
for path, needle in checks:
    found = needle in open(path, encoding='utf-8').read()
    print(f'{"OK" if found else "MISSING"} — {path}: {needle}')
