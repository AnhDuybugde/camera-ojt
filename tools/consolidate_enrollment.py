"""Copy legacy enrollment samples into the unified attendance database.

The legacy source remains untouched, so this operation can safely be repeated.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "apps" / "attendance"))


def main():
    from database.db import Database
    from camera_tracking.store.embeddings import EmbeddingStore

    source = ROOT / "var" / "enrollment.db"
    if not source.exists():
        print("No legacy enrollment database; nothing to consolidate.")
        return 0
    db = Database()
    old = EmbeddingStore(source)
    new = EmbeddingStore(db.path)
    copied = 0
    with old.connect() as old_conn, db.transaction() as conn:
        rows = old_conn.execute("""SELECT employee_id, source_hash, model_version,
            source_name, dimension, vector, metadata, updated_at FROM enrollment_samples""").fetchall()
        for row in rows:
            conn.execute("""INSERT OR IGNORE INTO enrollment_samples
                (employee_id, source_hash, model_version, source_name, dimension, vector, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", row)
            copied += conn.execute("SELECT changes()").fetchone()[0]
    print(f"Enrollment rows copied into {db.path.name}: {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
