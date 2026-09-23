"""Thread-safe SQLite access and automatic schema initialization."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Sequence

from config import settings
from auth.permissions import SYSTEM, require_owner, require_permission
from utils.helpers import iso_now, validate_employee_id


WORK_STATUSES = frozenset({"ON", "WFH", "OFF"})
WORK_SESSIONS = frozenset({"MORNING", "AFTERNOON"})


class Database:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or settings.database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=20, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=20000")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock, self.connect() as conn:
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
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    department TEXT NOT NULL DEFAULT '',
                    position TEXT NOT NULL DEFAULT '',
                    email TEXT NOT NULL DEFAULT '',
                    phone TEXT NOT NULL DEFAULT '',
                    created_date TEXT NOT NULL,
                    face_embedding BLOB,
                    embedding_dim INTEGER,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT NOT NULL,
                    employee_name TEXT NOT NULL,
                    department TEXT NOT NULL DEFAULT '',
                    date TEXT NOT NULL,
                    check_in TEXT,
                    check_out TEXT,
                    status TEXT NOT NULL DEFAULT 'ON_TIME',
                    sync_status TEXT NOT NULL DEFAULT 'PENDING',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(employee_id, date),
                    FOREIGN KEY(employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS enrollment_samples (
                    employee_id TEXT NOT NULL, source_hash TEXT NOT NULL,
                    model_version TEXT NOT NULL, source_name TEXT NOT NULL,
                    dimension INTEGER NOT NULL, vector BLOB NOT NULL,
                    metadata TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(employee_id, source_hash, model_version)
                );
                CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
                CREATE INDEX IF NOT EXISTS idx_attendance_sync ON attendance(sync_status);
                CREATE TABLE IF NOT EXISTS work_schedules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT NOT NULL,
                    work_date TEXT NOT NULL,
                    work_session TEXT NOT NULL CHECK(work_session IN ('MORNING', 'AFTERNOON')),
                    work_status TEXT NOT NULL CHECK(work_status IN ('ON', 'WFH', 'OFF')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(employee_id, work_date, work_session),
                    FOREIGN KEY(employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS auth_settings (
                    role TEXT PRIMARY KEY CHECK(role IN ('EMPLOYEE', 'ADMIN')),
                    password_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    employee_id TEXT NOT NULL DEFAULT '',
                    old_values TEXT NOT NULL DEFAULT '{}',
                    new_values TEXT NOT NULL DEFAULT '{}',
                    reason TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp
                    ON audit_logs(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_logs_employee
                    ON audit_logs(employee_id, timestamp DESC);
                """
            )
            self._migrate_work_schedule_sessions(conn)
            from camera_tracking.store.event_log import initialize_events
            initialize_events(conn)
            from camera_tracking.store.replication import TABLES, ENROLLMENT_TABLES, initialize_replication
            initialize_replication(conn, {**TABLES, **ENROLLMENT_TABLES})

    @staticmethod
    def _migrate_work_schedule_sessions(conn: sqlite3.Connection) -> None:
        """Expand legacy daily schedules into morning and afternoon rows."""
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(work_schedules)").fetchall()
        }
        if "work_session" not in columns:
            conn.execute("ALTER TABLE work_schedules RENAME TO work_schedules_daily_backup")
            conn.execute(
                """CREATE TABLE work_schedules (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       employee_id TEXT NOT NULL,
                       work_date TEXT NOT NULL,
                       work_session TEXT NOT NULL CHECK(work_session IN ('MORNING', 'AFTERNOON')),
                       work_status TEXT NOT NULL CHECK(work_status IN ('ON', 'WFH', 'OFF')),
                       created_at TEXT NOT NULL,
                       updated_at TEXT NOT NULL,
                       UNIQUE(employee_id, work_date, work_session),
                       FOREIGN KEY(employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
                   )"""
            )
            conn.execute(
                """INSERT INTO work_schedules
                       (employee_id, work_date, work_session, work_status, created_at, updated_at)
                   SELECT employee_id, work_date, 'MORNING', work_status, created_at, updated_at
                   FROM work_schedules_daily_backup
                   UNION ALL
                   SELECT employee_id, work_date, 'AFTERNOON', work_status, created_at, updated_at
                   FROM work_schedules_daily_backup"""
            )
            conn.execute("DROP TABLE work_schedules_daily_backup")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_work_schedules_date ON work_schedules(work_date)"
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_work_schedules_status_date
               ON work_schedules(work_status, work_date, work_session)"""
        )

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self.transaction() as conn:
            cursor = conn.execute(sql, params)
            return int(cursor.lastrowid or cursor.rowcount)

    def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def add_employee(self, data: dict[str, Any], *, actor_role: str = SYSTEM) -> None:
        require_permission(actor_role, "employee.add")
        employee_id = validate_employee_id(str(data["employee_id"]))
        name = str(data["full_name"]).strip()
        if not name:
            raise ValueError("Full name is required.")
        now = iso_now()
        self.execute(
            """INSERT INTO employees
               (employee_id, full_name, department, position, email, phone, created_date, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (employee_id, name, data.get("department", "").strip(), data.get("position", "").strip(),
             data.get("email", "").strip(), data.get("phone", "").strip(), now[:10], now),
        )

    def update_employee(
        self, employee_id: str, data: dict[str, Any], *, actor_role: str
    ) -> None:
        require_permission(actor_role, "employee.update")
        employee_id = validate_employee_id(employee_id)
        name = str(data["full_name"]).strip()
        if not name:
            raise ValueError("Full name is required.")
        self.execute(
            """UPDATE employees SET full_name=?, department=?, position=?, email=?, phone=?, updated_at=?
               WHERE employee_id=?""",
            (name, data.get("department", "").strip(), data.get("position", "").strip(),
             data.get("email", "").strip(), data.get("phone", "").strip(), iso_now(), employee_id),
        )

    def delete_employee(self, employee_id: str, *, actor_role: str) -> None:
        require_permission(actor_role, "employee.delete")
        self.execute("DELETE FROM employees WHERE employee_id=?", (validate_employee_id(employee_id),))

    def get_employee(self, employee_id: str) -> dict[str, Any] | None:
        return self.fetch_one(
            """SELECT employee_id, full_name, department, position, email, phone, created_date,
                      face_embedding IS NOT NULL AS has_face
               FROM employees WHERE employee_id=?""", (employee_id,)
        )

    def list_employees(self) -> list[dict[str, Any]]:
        return self.fetch_all(
            """SELECT employee_id, full_name, department, position, email, phone, created_date,
                      face_embedding IS NOT NULL AS has_face
               FROM employees ORDER BY full_name"""
        )

    def save_embedding(self, employee_id: str, blob: bytes, dimension: int, *, actor_role: str = SYSTEM, actor_employee_id: str | None = None) -> None:
        require_owner(actor_role, employee_id, actor_employee_id)
        with self.transaction() as conn:
            row = conn.execute("SELECT face_embedding FROM employees WHERE employee_id=?", (employee_id,)).fetchone()
            if not row:
                raise ValueError("Employee does not exist.")
            if actor_role == "EMPLOYEE" and row["face_embedding"] is not None:
                raise PermissionError("Đăng ký lại khuôn mặt cần quản lý thực hiện.")
            conn.execute("UPDATE employees SET face_embedding=?, embedding_dim=?, updated_at=? WHERE employee_id=?", (blob, dimension, iso_now(), employee_id))
            conn.execute("INSERT INTO audit_logs(timestamp,role,action,employee_id) VALUES (?,?,?,?)", (iso_now(), actor_role, "FACE_REGISTER", employee_id))

    def list_embeddings(self) -> list[dict[str, Any]]:
        return self.fetch_all(
            """SELECT employee_id, full_name, department, face_embedding, embedding_dim
               FROM employees WHERE face_embedding IS NOT NULL"""
        )

    def save_embedding_samples(self, employee_id: str, vectors: list, *,
                               actor_role: str = SYSTEM, actor_employee_id: str | None = None):
        """Persist all accepted samples using the configured backend model space."""
        import numpy as np
        from camera_tracking.config import load_config
        from camera_tracking.store.embeddings import EmbeddingStore
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        cfg = load_config(root / "config/default.yaml").face
        samples = [np.asarray(v, dtype=np.float32).ravel() for v in vectors]
        if not samples or len(samples) > 100 or any(
            v.size != 512 or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-12 for v in samples
        ):
            raise ValueError("Enrollment requires 1–100 valid 512-dimensional samples")
        samples = [v / np.linalg.norm(v) for v in samples]
        center = np.mean(samples, axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-12)
        require_owner(actor_role, employee_id, actor_employee_id)
        store = EmbeddingStore(self.path)
        version = f"{cfg.model_pack}:insightface-0.7:aligned-v1:det{cfg.det_size}"
        with self.transaction() as conn:
            row = conn.execute("SELECT face_embedding FROM employees WHERE employee_id=?", (employee_id,)).fetchone()
            if not row:
                raise ValueError("Employee does not exist.")
            if actor_role == "EMPLOYEE" and row["face_embedding"] is not None:
                raise PermissionError("Đăng ký lại khuôn mặt cần quản lý thực hiện.")
            conn.execute("UPDATE employees SET face_embedding=?, embedding_dim=?, updated_at=? WHERE employee_id=?",
                         (center.tobytes(), center.size, iso_now(), employee_id))
            conn.execute("INSERT INTO audit_logs(timestamp,role,action,employee_id) VALUES (?,?,?,?)",
                         (iso_now(), actor_role, "FACE_REGISTER", employee_id))
            store.replace(employee_id, version, samples, connection=conn)

    def list_attendance(
        self, date_from: str | None = None, date_to: str | None = None,
        employee_id: str | None = None, department: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if date_from:
            clauses.append("date >= ?"); params.append(date_from)
        if date_to:
            clauses.append("date <= ?"); params.append(date_to)
        if employee_id:
            clauses.append("employee_id = ?"); params.append(employee_id)
        if department:
            clauses.append("department = ?"); params.append(department)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return self.fetch_all("SELECT * FROM attendance" + where + " ORDER BY date DESC, check_in DESC", params)

    def get_attendance(self, attendance_id: int) -> dict[str, Any] | None:
        return self.fetch_one("SELECT * FROM attendance WHERE id=?", (attendance_id,))

    def list_audit_logs(self, limit: int = 500) -> list[dict[str, Any]]:
        return self.fetch_all(
            "SELECT * FROM audit_logs ORDER BY timestamp DESC, id DESC LIMIT ?",
            (limit,),
        )

    def get_work_schedule(
        self, employee_id: str, work_date: str, work_session: str = "MORNING"
    ) -> dict[str, Any] | None:
        employee_id = validate_employee_id(employee_id)
        work_date = date.fromisoformat(work_date).isoformat()
        work_session = work_session.strip().upper()
        if work_session not in WORK_SESSIONS:
            raise ValueError("Work session must be MORNING or AFTERNOON.")
        return self.fetch_one(
            """SELECT * FROM work_schedules
               WHERE employee_id=? AND work_date=? AND work_session=?""",
            (employee_id, work_date, work_session),
        )

    def list_work_schedules(
        self,
        date_from: str,
        date_to: str,
        employee_id: str | None = None,
        department: str | None = None,
    ) -> list[dict[str, Any]]:
        date_from = date.fromisoformat(date_from).isoformat()
        date_to = date.fromisoformat(date_to).isoformat()
        clauses = ["ws.work_date >= ?", "ws.work_date <= ?"]
        params: list[Any] = [date_from, date_to]
        if employee_id:
            clauses.append("ws.employee_id = ?")
            params.append(validate_employee_id(employee_id))
        if department:
            clauses.append("e.department = ?")
            params.append(department)
        return self.fetch_all(
            """SELECT ws.*, e.full_name, e.department
               FROM work_schedules ws
               JOIN employees e ON e.employee_id = ws.employee_id
               WHERE """ + " AND ".join(clauses) +
            " ORDER BY e.full_name, ws.work_date, ws.work_session DESC",
            params,
        )

    def save_work_schedule(
        self,
        employee_id: str,
        work_date: str,
        work_status: str,
        work_session: str = "MORNING",
    ) -> None:
        self.save_work_schedules([(employee_id, work_date, work_session, work_status)])

    def save_work_schedules(
        self, entries: Sequence[tuple[str, str, str, str | None]], *, actor_role: str = SYSTEM, actor_employee_id: str | None = None
    ) -> None:
        """Atomically upsert schedule cells; a None status clears that cell."""
        normalized: list[tuple[str, str, str, str | None]] = []
        for employee_id, work_date, work_session, work_status in entries:
            require_permission(actor_role, "schedule.update")
            require_owner(actor_role, employee_id, actor_employee_id)
            clean_id = validate_employee_id(employee_id)
            clean_date = date.fromisoformat(work_date).isoformat()
            clean_session = work_session.strip().upper()
            if clean_session not in WORK_SESSIONS:
                raise ValueError("Work session must be MORNING or AFTERNOON.")
            clean_status = work_status.strip().upper() if work_status is not None else None
            if clean_status is not None and clean_status not in WORK_STATUSES:
                raise ValueError("Work status must be ON, WFH or OFF.")
            normalized.append((clean_id, clean_date, clean_session, clean_status))

        now = iso_now()
        with self.transaction() as conn:
            conn.execute("INSERT INTO audit_logs(timestamp,role,action,employee_id) VALUES (?,?,?,?)", (now, actor_role, "SCHEDULE_UPDATE", actor_employee_id or ""))
            for employee_id, work_date, work_session, work_status in normalized:
                if work_status is None:
                    conn.execute(
                        """DELETE FROM work_schedules
                           WHERE employee_id=? AND work_date=? AND work_session=?""",
                        (employee_id, work_date, work_session),
                    )
                    continue
                conn.execute(
                    """INSERT INTO work_schedules
                       (employee_id, work_date, work_session, work_status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(employee_id, work_date, work_session) DO UPDATE SET
                           work_status=excluded.work_status,
                           updated_at=excluded.updated_at""",
                    (employee_id, work_date, work_session, work_status, now, now),
                )

    def pending_attendance(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.fetch_all(
            "SELECT * FROM attendance WHERE sync_status != 'SYNCED' ORDER BY updated_at LIMIT ?", (limit,)
        )

    def mark_synced(self, attendance_id: int) -> None:
        self.execute("UPDATE attendance SET sync_status='SYNCED' WHERE id=?", (attendance_id,))

    def mark_sync_error(self, attendance_id: int) -> None:
        self.execute("UPDATE attendance SET sync_status='ERROR' WHERE id=?", (attendance_id,))
