"""Spatial event repository; PostgreSQL in production, SQLite for local use."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine

from database.postgres import ConnectionAdapter, metadata


def local_now() -> datetime:
    return datetime.now().astimezone()


def iso(value: datetime | None = None) -> str:
    return (value or local_now()).isoformat(timespec="seconds")


class SpatialRepository:
    def __init__(self, path: str | Path, *, database_url: str = "") -> None:
        self.database_url = database_url.strip()
        self.is_postgres = self.database_url.startswith("postgresql+psycopg://")
        if self.database_url and not self.is_postgres:
            raise ValueError("Spatial database_url must use postgresql+psycopg://")
        self._engine = create_engine(self.database_url, pool_pre_ping=True) if self.is_postgres else None
        self.path = Path(path)
        if not self.is_postgres:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.initialize()

    def connect(self):
        if self.is_postgres:
            return self._postgres_connection()
        conn = sqlite3.connect(self.path, timeout=20, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=20000")
        return conn

    @contextmanager
    def _postgres_connection(self):
        assert self._engine is not None
        with self._engine.connect() as conn:
            yield ConnectionAdapter(conn)

    @contextmanager
    def transaction(self) -> Iterator[object]:
        if self.is_postgres:
            assert self._engine is not None
            with self._engine.begin() as conn:
                yield ConnectionAdapter(conn)
            return
        with self._lock, self.connect() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def initialize(self) -> None:
        if self.is_postgres:
            assert self._engine is not None
            metadata.create_all(self._engine)
            return
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
                CREATE TABLE IF NOT EXISTS activity_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT,
                    employee_name TEXT NOT NULL DEFAULT 'Chưa xác định',
                    camera_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    activity_type TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_seconds REAL,
                    confidence REAL NOT NULL,
                    zone TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_activity_events_time
                    ON activity_events(started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_activity_events_employee
                    ON activity_events(employee_id, started_at DESC);
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

    def transition_activity(
        self, camera_id: str, track_id: int, employee_id: str | None,
        employee_name: str, activity_type: str, confidence: float,
        zone: str | None, at: datetime | None = None,
    ) -> None:
        """Close the previous event and create exactly one row for a transition."""
        changed_at = at or local_now()
        changed_iso = iso(changed_at)
        with self.transaction() as conn:
            open_row = conn.execute(
                """SELECT id,started_at,activity_type FROM activity_events
                   WHERE camera_id=? AND track_id=? AND ended_at IS NULL
                   ORDER BY started_at DESC LIMIT 1""",
                (camera_id, track_id),
            ).fetchone()
            if open_row and open_row["activity_type"] == activity_type:
                conn.execute(
                    """UPDATE activity_events SET employee_id=?,employee_name=?,confidence=?,zone=?
                       WHERE id=?""",
                    (employee_id, employee_name, float(confidence), zone, open_row["id"]),
                )
                return
            if open_row:
                started_at = datetime.fromisoformat(str(open_row["started_at"]))
                if started_at.tzinfo is None and changed_at.tzinfo is not None:
                    started_at = started_at.replace(tzinfo=changed_at.tzinfo)
                duration = max(0.0, (changed_at - started_at).total_seconds())
                conn.execute(
                    "UPDATE activity_events SET ended_at=?,duration_seconds=? WHERE id=?",
                    (changed_iso, duration, open_row["id"]),
                )
            # Short UNKNOWN flicker is useful in live UI but not useful storage.
            if activity_type != "UNKNOWN":
                conn.execute(
                    """INSERT INTO activity_events
                       (employee_id,employee_name,camera_id,track_id,activity_type,started_at,
                        ended_at,duration_seconds,confidence,zone,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (employee_id, employee_name, camera_id, track_id, activity_type,
                     changed_iso, None, None, float(confidence), zone, changed_iso),
                )

    def close_activity(self, camera_id: str, track_id: int, at: datetime | None = None) -> None:
        ended_at = at or local_now()
        with self.transaction() as conn:
            row = conn.execute(
                """SELECT id,started_at FROM activity_events
                   WHERE camera_id=? AND track_id=? AND ended_at IS NULL
                   ORDER BY started_at DESC LIMIT 1""",
                (camera_id, track_id),
            ).fetchone()
            if not row:
                return
            started_at = datetime.fromisoformat(str(row["started_at"]))
            if started_at.tzinfo is None and ended_at.tzinfo is not None:
                started_at = started_at.replace(tzinfo=ended_at.tzinfo)
            conn.execute(
                "UPDATE activity_events SET ended_at=?,duration_seconds=? WHERE id=?",
                (iso(ended_at), max(0.0, (ended_at - started_at).total_seconds()), row["id"]),
            )

    def bind_activity_identity(
        self, camera_id: str, track_id: int, employee_id: str, employee_name: str,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """UPDATE activity_events SET employee_id=?,employee_name=?
                   WHERE camera_id=? AND track_id=? AND ended_at IS NULL""",
                (employee_id, employee_name, camera_id, track_id),
            )

    def activity_rows(self, since: datetime, limit: int = 1000) -> list[dict]:
        return self._rows(
            """SELECT * FROM activity_events WHERE started_at>=?
               ORDER BY started_at DESC LIMIT ?""", (iso(since), limit),
        )

    def _rows(self, query: str, params: tuple) -> list[dict]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]
