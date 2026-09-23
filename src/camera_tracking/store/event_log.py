"""Durable observations and transactional outbox shared by all producers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4


def initialize_events(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS camera_events (
            event_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL DEFAULT 1,
            employee_id TEXT, camera_id TEXT NOT NULL, session_id TEXT NOT NULL,
            kind TEXT NOT NULL, observed_at TEXT NOT NULL,
            payload TEXT NOT NULL, needs_review INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS camera_events_time ON camera_events(observed_at);
        CREATE TABLE IF NOT EXISTS event_outbox (
            event_id TEXT PRIMARY KEY REFERENCES camera_events(event_id),
            attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL NOT NULL DEFAULT 0,
            last_error TEXT, delivered_at TEXT
        );
    """)


def append_event(conn, *, kind, observed_at, camera_id, session_id,
                 employee_id=None, payload=None, event_id=None, needs_review=False):
    if observed_at.tzinfo is None:
        raise ValueError("Observation timestamps must include a timezone")
    event_id = event_id or str(uuid4())
    values = (event_id, employee_id, camera_id, session_id, kind,
              observed_at.astimezone(timezone.utc).isoformat(),
              json.dumps(payload or {}, ensure_ascii=False, sort_keys=True), int(needs_review))
    inserted = conn.execute("""INSERT OR IGNORE INTO camera_events
        (event_id, employee_id, camera_id, session_id, kind, observed_at, payload, needs_review)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", values).rowcount
    if inserted:
        conn.execute("INSERT INTO event_outbox(event_id) VALUES (?)", (event_id,))
    return event_id, bool(inserted)


class EventSync:
    """At-least-once delivery; the remote event_id must be a unique key.

    Network calls run outside SQLite transactions. Errors never include provider
    exception text, which can contain credentials or private payloads.
    """
    def __init__(self, db, send, *, clock=None):
        import time
        self.db, self.send, self.clock = db, send, clock or time.time

    def flush(self, limit=100):
        rows = self.db.fetch_all("""SELECT e.*, o.attempts FROM event_outbox o
            JOIN camera_events e USING(event_id)
            WHERE o.delivered_at IS NULL AND o.next_attempt<=?
            ORDER BY e.observed_at LIMIT ?""", (self.clock(), limit))
        delivered = 0
        for row in rows:
            attempts = row.pop("attempts")
            row["payload"] = json.loads(row["payload"])
            row["needs_review"] = bool(row["needs_review"])
            try:
                self.send(row)
            except Exception as error:
                self.db.execute("""UPDATE event_outbox SET attempts=attempts+1,
                    next_attempt=?, last_error=? WHERE event_id=?""",
                    (self.clock() + min(3600, 2 ** min(attempts + 1, 12)),
                     type(error).__name__, row["event_id"]))
            else:
                self.db.execute("""UPDATE event_outbox SET delivered_at=?, last_error=NULL
                    WHERE event_id=?""", (datetime.now(timezone.utc).isoformat(), row["event_id"]))
                delivered += 1
        return delivered
