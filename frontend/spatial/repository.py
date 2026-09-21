"""Separate SQLite store for spatial analytics; attendance data is never mutated."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator


def local_now() -> datetime:
    return datetime.now().astimezone()


def iso(value: datetime | None = None) -> str:
    return (value or local_now()).isoformat(timespec="seconds")


class SpatialRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=20, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=20000")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self.connect() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def initialize(self) -> None:
        with self.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS density_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_at TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    zone_id TEXT NOT NULL,
                    people_count INTEGER NOT NULL CHECK(people_count >= 0)
                );
                CREATE INDEX IF NOT EXISTS idx_density_time
                    ON density_samples(captured_at, camera_id, zone_id);
                CREATE TABLE IF NOT EXISTS movement_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_time TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    employee_id TEXT,
                    employee_name TEXT NOT NULL DEFAULT 'Chưa xác định',
                    from_zone TEXT,
                    to_zone TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_movement_time
                    ON movement_events(event_time DESC);
                CREATE TABLE IF NOT EXISTS zone_visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    employee_id TEXT,
                    employee_name TEXT NOT NULL DEFAULT 'Chưa xác định',
                    zone_id TEXT NOT NULL,
                    entered_at TEXT NOT NULL,
                    exited_at TEXT NOT NULL,
                    dwell_seconds REAL NOT NULL CHECK(dwell_seconds >= 0)
                );
                CREATE INDEX IF NOT EXISTS idx_visits_time ON zone_visits(entered_at DESC);
                """
            )

    def record_density(self, camera_id: str, counts: dict[str, int], at: datetime | None = None) -> None:
        captured_at = iso(at)
        with self.transaction() as conn:
            conn.executemany(
                "INSERT INTO density_samples(captured_at,camera_id,zone_id,people_count) VALUES(?,?,?,?)",
                [(captured_at, camera_id, zone_id, int(count)) for zone_id, count in counts.items()],
            )

    def record_transition(
        self, camera_id: str, track_id: int, employee_id: str | None,
        employee_name: str, from_zone: str | None, to_zone: str | None,
        at: datetime | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO movement_events
                   (event_time,camera_id,track_id,employee_id,employee_name,from_zone,to_zone)
                   VALUES(?,?,?,?,?,?,?)""",
                (iso(at), camera_id, track_id, employee_id, employee_name, from_zone, to_zone),
            )

    def record_visit(
        self, camera_id: str, track_id: int, employee_id: str | None,
        employee_name: str, zone_id: str, entered_at: datetime, exited_at: datetime,
    ) -> None:
        dwell = max(0.0, (exited_at - entered_at).total_seconds())
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO zone_visits
                   (camera_id,track_id,employee_id,employee_name,zone_id,entered_at,exited_at,dwell_seconds)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (camera_id, track_id, employee_id, employee_name, zone_id,
                 iso(entered_at), iso(exited_at), dwell),
            )

    def density_rows(self, since: datetime) -> list[dict]:
        return self._rows(
            "SELECT * FROM density_samples WHERE captured_at>=? ORDER BY captured_at", (iso(since),)
        )

    def transition_rows(self, since: datetime, limit: int = 250) -> list[dict]:
        return self._rows(
            """SELECT * FROM movement_events WHERE event_time>=?
               ORDER BY event_time DESC LIMIT ?""", (iso(since), limit),
        )

    def visit_rows(self, since: datetime) -> list[dict]:
        return self._rows(
            "SELECT * FROM zone_visits WHERE entered_at>=? ORDER BY entered_at DESC", (iso(since),)
        )

    def _rows(self, query: str, params: tuple) -> list[dict]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]
