"""SQLite queue cho write Supabase: offline van chay, co mang flush sau."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable
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

_COALESCED_KEYS = {
    "room_status": ("date", "global_id"),
    "current_state": ("employee_id",),
}


class WriteQueue:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def push(self, kind: str, payload: dict) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            key_fields = _COALESCED_KEYS.get(kind)
            if key_fields and all(field in payload for field in key_fields):
                where = " AND ".join(f"json_extract(payload, '$.{field}') = ?"
                                     for field in key_fields)
                values = [kind, *(payload[field] for field in key_fields)]
                row = conn.execute(
                    f"SELECT id FROM pending_writes WHERE kind = ? AND {where} "
                    "ORDER BY id DESC LIMIT 1",
                    values,
                ).fetchone()
                if row is not None:
                    conn.execute(
                        "UPDATE pending_writes SET payload = ?, created_at = ?, "
                        "attempts = 0 WHERE id = ?",
                        (json.dumps(payload, ensure_ascii=False), time.time(), row[0]),
                    )
                    conn.commit()
                    return int(row[0])
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


class WriteQueueWorker:
    """Flush queued writes off the inference thread."""

    def __init__(
        self,
        queue: WriteQueue,
        flush: Callable[[], object],
        interval_s: float = 1.0,
    ) -> None:
        self.queue = queue
        self.flush = flush
        self.interval_s = max(0.1, interval_s)
        self._stop = threading.Event()
        self._flush_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="write-queue-worker", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.flush_once()

    def flush_once(self) -> object:
        with self._flush_lock:
            return self.flush()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 2))
        self.flush_once()
