"""SQLite queue cho write Supabase: offline van chay, co mang flush sau."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_writes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  created_at REAL NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0
);
"""


class WriteQueue:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def push(self, kind: str, payload: dict) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.execute(
                "INSERT INTO pending_writes (kind, payload, created_at) VALUES (?, ?, ?)",
                (kind, json.dumps(payload, ensure_ascii=False), time.time()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def peek(self, limit: int = 100) -> list[tuple[int, str, dict, int]]:
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT id, kind, payload, attempts FROM pending_writes "
                "ORDER BY id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [(r[0], r[1], json.loads(r[2]), r[3]) for r in rows]

    def ack(self, row_id: int) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM pending_writes WHERE id = ?", (row_id,))
            conn.commit()

    def bump(self, row_id: int) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE pending_writes SET attempts = attempts + 1 WHERE id = ?",
                (row_id,),
            )
            conn.commit()

    def __len__(self) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            (count,) = conn.execute("SELECT COUNT(*) FROM pending_writes").fetchone()
        return int(count)
