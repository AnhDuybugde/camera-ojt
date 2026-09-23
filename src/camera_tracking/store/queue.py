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
  coalesce_key TEXT,
  created_at REAL NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0
);
"""

_COALESCED_KEYS = {
    "person": ("person_id",),
    "room_status": ("date", "global_id"),
    "current_state": ("employee_id",),
}


class WriteQueue:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(_SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(pending_writes)")}
            if "coalesce_key" not in columns:
                conn.execute("ALTER TABLE pending_writes ADD COLUMN coalesce_key TEXT")
            for kind, fields in _COALESCED_KEYS.items():
                rows = conn.execute("SELECT id,payload FROM pending_writes WHERE kind=? AND coalesce_key IS NULL",
                                    (kind,)).fetchall()
                for row_id, raw in rows:
                    payload = json.loads(raw)
                    if all(field in payload for field in fields):
                        key = json.dumps([payload[field] for field in fields],
                                         ensure_ascii=False, separators=(",", ":"))
                        conn.execute("UPDATE pending_writes SET coalesce_key=? WHERE id=?", (key, row_id))
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_coalesce "
                         "ON pending_writes(kind,coalesce_key,id DESC)")
            conn.commit()

    def push(self, kind: str, payload: dict) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            key_fields = _COALESCED_KEYS.get(kind)
            if key_fields and all(field in payload for field in key_fields):
                coalesce_key = json.dumps([payload[field] for field in key_fields],
                                          ensure_ascii=False, separators=(",", ":"))
                row = conn.execute(
                    "SELECT id FROM pending_writes WHERE kind = ? AND coalesce_key = ? "
                    "ORDER BY id DESC LIMIT 1",
                    (kind, coalesce_key),
                ).fetchone()
                if row is not None:
                    conn.execute(
                        "UPDATE pending_writes SET payload = ?, coalesce_key = ?, created_at = ?, "
                        "attempts = 0 WHERE id = ?",
                        (json.dumps(payload, ensure_ascii=False), coalesce_key, time.time(), row[0]),
                    )
                    conn.commit()
                    return int(row[0])
            cur = conn.execute(
                "INSERT INTO pending_writes (kind, payload, coalesce_key, created_at) VALUES (?, ?, ?, ?)",
                (kind, json.dumps(payload, ensure_ascii=False),
                 coalesce_key if key_fields and all(field in payload for field in key_fields) else None,
                 time.time()),
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

    def peek_kind(self, kind: str, limit: int = 100) -> list[tuple[int, str, dict, int]]:
        """Read one consumer's rows without being blocked by other kinds."""
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT id, kind, payload, attempts FROM pending_writes "
                "WHERE kind = ? ORDER BY id ASC LIMIT ?",
                (kind, limit),
            ).fetchall()
        return [(r[0], r[1], json.loads(r[2]), r[3]) for r in rows]

    def ack(self, row_id: int, expected_payload: dict | None = None) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            if expected_payload is None:
                conn.execute("DELETE FROM pending_writes WHERE id = ?", (row_id,))
            else:
                conn.execute("DELETE FROM pending_writes WHERE id = ? AND payload = ?",
                    (row_id, json.dumps(expected_payload, ensure_ascii=False)))
            conn.commit()

    def quarantine(self, row_id: int, reason: str) -> None:
        """Retain rejected payloads for reconciliation instead of dropping data."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS rejected_writes (
                id INTEGER PRIMARY KEY, kind TEXT, payload TEXT, created_at REAL,
                attempts INTEGER, reason TEXT NOT NULL)""")
            conn.execute("INSERT OR REPLACE INTO rejected_writes "
                         "SELECT id,kind,payload,created_at,attempts,? FROM pending_writes WHERE id=?",
                         (reason, row_id))
            conn.execute("DELETE FROM pending_writes WHERE id=?", (row_id,))

    def restore_mapped(self, mappings: dict[str, str]) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='rejected_writes'").fetchone():
                return 0
            rows = conn.execute("SELECT id,payload FROM rejected_writes WHERE reason='unmapped_employee'").fetchall()
            restored = 0
            for row_id, payload in rows:
                if json.loads(payload).get("person_id") not in mappings:
                    continue
                from_payload = json.loads(payload)
                # The row kind is fetched separately to preserve legacy ids.
                rejected = conn.execute("SELECT kind FROM rejected_writes WHERE id=?", (row_id,)).fetchone()
                row_kind = rejected[0] if rejected else ""
                key_fields = _COALESCED_KEYS.get(row_kind)
                coalesce_key = (json.dumps([from_payload[field] for field in key_fields],
                    ensure_ascii=False, separators=(",", ":"))
                    if key_fields and all(field in from_payload for field in key_fields) else None)
                conn.execute("INSERT OR IGNORE INTO pending_writes "
                    "(id,kind,payload,coalesce_key,created_at,attempts) "
                    "SELECT id,kind,payload,?,created_at,attempts FROM rejected_writes WHERE id=?",
                    (coalesce_key, row_id))
                conn.execute("DELETE FROM rejected_writes WHERE id=?", (row_id,))
                restored += 1
            return restored

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
