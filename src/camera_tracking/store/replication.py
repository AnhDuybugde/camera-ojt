"""Versioned record replication with a SQLite transactional outbox.

Only declared tables/columns participate. Concurrent cloud edits are preserved
as conflicts; offline edits never silently overwrite newer cloud revisions.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

TABLES = {
    "employees": (("employee_id",), ("employee_id", "full_name", "department", "position",
        "email", "phone", "created_date", "updated_at")),
    "work_schedules": (("employee_id", "work_date", "work_session"),
        ("employee_id", "work_date", "work_session", "work_status", "created_at", "updated_at")),
    "attendance": (("employee_id", "date"), ("employee_id", "employee_name", "department",
        "date", "check_in", "check_out", "status", "created_at", "updated_at")),
    "audit_logs": (("id",), ("id", "timestamp", "role", "action", "employee_id",
        "old_values", "new_values", "reason")),
}
ENROLLMENT_TABLES = {
    "enrollment_samples": (("employee_id", "source_hash", "model_version"),
        ("employee_id", "source_hash", "model_version", "source_name", "dimension",
         "vector", "metadata", "updated_at")),
}


def initialize_replication(conn, tables=TABLES):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sync_control (id INTEGER PRIMARY KEY, applying INTEGER NOT NULL);
        INSERT OR IGNORE INTO sync_control VALUES (1,0);
        CREATE TABLE IF NOT EXISTS record_versions (
            kind TEXT, record_key TEXT, revision INTEGER NOT NULL,
            PRIMARY KEY(kind, record_key));
        CREATE TABLE IF NOT EXISTS record_outbox (
            kind TEXT, record_key TEXT, payload TEXT NOT NULL, deleted INTEGER NOT NULL,
            base_revision INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(kind, record_key));
        CREATE TABLE IF NOT EXISTS record_conflicts (
            kind TEXT, record_key TEXT, remote TEXT NOT NULL,
            PRIMARY KEY(kind, record_key));
        CREATE TABLE IF NOT EXISTS sync_cursor (id INTEGER PRIMARY KEY, sequence INTEGER NOT NULL);
        INSERT OR IGNORE INTO sync_cursor VALUES (1,0);
    """)
    for table, (keys, columns) in tables.items():
        for action, prefix, deleted in (("INSERT", "NEW", 0), ("UPDATE", "NEW", 0), ("DELETE", "OLD", 1)):
            key = "json_array(" + ",".join(f"{prefix}.{k}" for k in keys) + ")"
            items = []
            for column in columns:
                value = f"hex({prefix}.vector)" if column == "vector" else f"{prefix}.{column}"
                items.extend((f"'{column}'", value))
            payload = "json_object(" + ",".join(items) + ")"
            conn.execute(f"""CREATE TRIGGER IF NOT EXISTS replicate_{table}_{action}
                AFTER {action} ON {table} WHEN (SELECT applying FROM sync_control WHERE id=1)=0
                BEGIN
                    INSERT INTO record_outbox(kind,record_key,payload,deleted,base_revision)
                    VALUES ('{table}',{key},{payload},{deleted},COALESCE(
                        (SELECT revision FROM record_versions WHERE kind='{table}' AND record_key={key}),0))
                    ON CONFLICT(kind,record_key) DO UPDATE SET
                        payload=excluded.payload,deleted=excluded.deleted;
                END""")


class SQLiteReplica:
    def __init__(self, path, tables=TABLES):
        self.path, self.tables = Path(path), tables

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def bootstrap(self):
        """Queue existing rows exactly once without modifying their timestamps."""
        with self.connect() as db:
            initialize_replication(db, self.tables)
            for table, (keys, columns) in self.tables.items():
                for row in db.execute(f"SELECT {','.join(columns)} FROM {table}").fetchall():
                    payload = dict(row)
                    if "vector" in payload:
                        payload["vector"] = payload["vector"].hex().upper()
                    key = json.dumps([payload[k] for k in keys], ensure_ascii=False, separators=(",", ":"))
                    db.execute("""INSERT OR IGNORE INTO record_outbox
                        SELECT ?,?,?,0,0 WHERE NOT EXISTS
                        (SELECT 1 FROM record_versions WHERE kind=? AND record_key=?)""",
                        (table, key, json.dumps(payload), table, key))

    def apply(self, db, record):
        table = record["kind"]
        if table not in self.tables:
            return
        keys, columns = self.tables[table]
        payload = record["payload"]
        if json.loads(record["record_key"]) != [payload[k] for k in keys]:
            raise ValueError("Cloud record key does not match its payload")
        db.execute("UPDATE sync_control SET applying=1 WHERE id=1")
        try:
            if record["deleted"]:
                db.execute(f"DELETE FROM {table} WHERE " + " AND ".join(f"{k}=?" for k in keys),
                           [payload[k] for k in keys])
            else:
                values = [bytes.fromhex(payload[c]) if c == "vector" else payload[c] for c in columns]
                updates = ",".join(f"{c}=excluded.{c}" for c in columns if c not in keys)
                db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES "
                    f"({','.join('?' for _ in columns)}) ON CONFLICT({','.join(keys)}) "
                    f"DO UPDATE SET {updates}", values)
            db.execute("INSERT INTO record_versions VALUES (?,?,?) ON CONFLICT(kind,record_key) "
                       "DO UPDATE SET revision=excluded.revision",
                       (table, record["record_key"], record["revision"]))
        finally:
            db.execute("UPDATE sync_control SET applying=0 WHERE id=1")

    def sync(self, remote, *, allowed_kinds=None):
        allowed = None if allowed_kinds is None else set(allowed_kinds)
        with self.connect() as db:
            rows = db.execute("""SELECT o.* FROM record_outbox o LEFT JOIN record_conflicts c
                USING(kind,record_key) WHERE c.kind IS NULL ORDER BY
                CASE o.kind WHEN 'employees' THEN 0 WHEN 'work_schedules' THEN 1 ELSE 2 END
                LIMIT 100""").fetchall()
        if allowed is not None:
            rows = [row for row in rows if row["kind"] in allowed]
        results = remote.compare_batch([dict(row) for row in rows]) if rows else []
        if len(results) != len(rows):
            raise ValueError("Cloud batch acknowledgement count mismatch")
        for row, result in zip(rows, results):
            record = result["record"]
            with self.connect() as db:
                if result["conflict"]:
                    db.execute("INSERT OR REPLACE INTO record_conflicts VALUES (?,?,?)",
                               (row["kind"], row["record_key"], json.dumps(record)))
                    continue
                db.execute("INSERT OR REPLACE INTO record_versions VALUES (?,?,?)",
                           (row["kind"], row["record_key"], record["revision"]))
                db.execute("DELETE FROM record_outbox WHERE kind=? AND record_key=? AND payload=? AND deleted=?",
                           (row["kind"], row["record_key"], row["payload"], row["deleted"]))
                db.execute("UPDATE record_outbox SET base_revision=? WHERE kind=? AND record_key=?",
                           (record["revision"], row["kind"], row["record_key"]))
        with self.connect() as db:
            cursor = db.execute("SELECT sequence FROM sync_cursor WHERE id=1").fetchone()[0]
        for record in remote.changes(cursor):
            with self.connect() as db:
                table, key = record["kind"], record["record_key"]
                if table in self.tables:
                    version = db.execute("SELECT revision FROM record_versions WHERE kind=? AND record_key=?",
                                         (table, key)).fetchone()
                    pending = db.execute("SELECT 1 FROM record_outbox WHERE kind=? AND record_key=?",
                                         (table, key)).fetchone()
                    if not version or version[0] < record["revision"]:
                        if pending:
                            db.execute("INSERT OR REPLACE INTO record_conflicts VALUES (?,?,?)",
                                       (table, key, json.dumps(record)))
                        else:
                            self.apply(db, record)
                db.execute("UPDATE sync_cursor SET sequence=? WHERE id=1", (record["change_seq"],))

    def resolve(self, kind, key, choice):
        if choice not in {"local", "cloud"}:
            raise ValueError("Choose local or cloud")
        with self.connect() as db:
            row = db.execute("SELECT remote FROM record_conflicts WHERE kind=? AND record_key=?", (kind, key)).fetchone()
            if row is None:
                raise ValueError("Conflict no longer exists")
            record = json.loads(row[0])
            if choice == "cloud":
                self.apply(db, record)
                db.execute("DELETE FROM record_outbox WHERE kind=? AND record_key=?", (kind, key))
            else:
                db.execute("UPDATE record_outbox SET base_revision=? WHERE kind=? AND record_key=?",
                           (record["revision"], kind, key))
            db.execute("DELETE FROM record_conflicts WHERE kind=? AND record_key=?", (kind, key))
