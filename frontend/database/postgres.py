"""Central PostgreSQL schema and SQLAlchemy compatibility adapter.

The application still works with ISO strings. PostgreSQL stores dates and
timestamps in native types; the adapter converts at the repository boundary.
"""
from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from sqlalchemy import (
    CheckConstraint, Column, Date, DateTime, Float, ForeignKey, Index, Integer,
    LargeBinary, MetaData, String, Table, Text, UniqueConstraint, text,
)
from sqlalchemy.engine import Connection


metadata = MetaData()

employees = Table(
    "employees", metadata,
    Column("employee_id", String(32), primary_key=True),
    Column("full_name", Text, nullable=False),
    Column("department", Text, nullable=False, server_default=""),
    Column("position", Text, nullable=False, server_default=""),
    Column("email", Text, nullable=False, server_default=""),
    Column("phone", Text, nullable=False, server_default=""),
    Column("created_date", Date, nullable=False),
    Column("face_embedding", LargeBinary),
    Column("embedding_dim", Integer),
    Column("updated_at", DateTime(timezone=False), nullable=False),
)

attendance = Table(
    "attendance", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("employee_id", String(32), ForeignKey("employees.employee_id", ondelete="CASCADE"), nullable=False),
    Column("employee_name", Text, nullable=False),
    Column("department", Text, nullable=False, server_default=""),
    Column("date", Date, nullable=False),
    Column("check_in", DateTime(timezone=False)),
    Column("check_out", DateTime(timezone=False)),
    Column("status", Text, nullable=False, server_default="ON_TIME"),
    Column("presence_status", Text, nullable=False, server_default="ABSENT"),
    Column("temporary_checkout_at", DateTime(timezone=False)),
    Column("sync_status", Text, nullable=False, server_default="PENDING"),
    Column("created_at", DateTime(timezone=False), nullable=False),
    Column("updated_at", DateTime(timezone=False), nullable=False),
    UniqueConstraint("employee_id", "date"),
)
Index("idx_attendance_date", attendance.c.date)
Index("idx_attendance_sync", attendance.c.sync_status)

work_schedules = Table(
    "work_schedules", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("employee_id", String(32), ForeignKey("employees.employee_id", ondelete="CASCADE"), nullable=False),
    Column("work_date", Date, nullable=False),
    Column("work_session", String(9), nullable=False),
    Column("work_status", String(3), nullable=False),
    Column("created_at", DateTime(timezone=False), nullable=False),
    Column("updated_at", DateTime(timezone=False), nullable=False),
    CheckConstraint("work_session IN ('MORNING', 'AFTERNOON')"),
    CheckConstraint("work_status IN ('ON', 'WFH', 'OFF')"),
    UniqueConstraint("employee_id", "work_date", "work_session"),
)
Index("idx_work_schedules_date", work_schedules.c.work_date)
Index("idx_work_schedules_status_date", work_schedules.c.work_status, work_schedules.c.work_date, work_schedules.c.work_session)

auth_settings = Table(
    "auth_settings", metadata,
    Column("role", String(8), primary_key=True),
    Column("password_hash", Text, nullable=False),
    Column("updated_at", DateTime(timezone=False), nullable=False),
    CheckConstraint("role IN ('EMPLOYEE', 'ADMIN')"),
)

audit_logs = Table(
    "audit_logs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=False), nullable=False),
    Column("role", Text, nullable=False),
    Column("action", Text, nullable=False),
    Column("employee_id", String(32), nullable=False, server_default=""),
    Column("old_values", Text, nullable=False, server_default="{}"),
    Column("new_values", Text, nullable=False, server_default="{}"),
    Column("reason", Text, nullable=False, server_default=""),
)
Index("idx_audit_logs_timestamp", audit_logs.c.timestamp.desc())
Index("idx_audit_logs_employee", audit_logs.c.employee_id, audit_logs.c.timestamp.desc())

employee_accounts = Table(
    "employee_accounts", metadata,
    Column("employee_id", String(32), ForeignKey("employees.employee_id", ondelete="CASCADE"), primary_key=True),
    Column("password_hash", Text, nullable=False),
    Column("must_change", Integer, nullable=False, server_default="1"),
    Column("failures", Integer, nullable=False, server_default="0"),
    Column("locked_until", Float, nullable=False, server_default="0"),
    Column("version", Integer, nullable=False, server_default="1"),
)

login_limits = Table(
    "login_limits", metadata,
    Column("account", Text, primary_key=True),
    Column("failures", Integer, nullable=False, server_default="0"),
    Column("locked_until", Float, nullable=False, server_default="0"),
)

density_samples = Table(
    "density_samples", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("captured_at", DateTime(timezone=True), nullable=False),
    Column("camera_id", Text, nullable=False),
    Column("zone_id", Text, nullable=False),
    Column("people_count", Integer, nullable=False),
    CheckConstraint("people_count >= 0"),
)
Index("idx_density_time", density_samples.c.captured_at, density_samples.c.camera_id, density_samples.c.zone_id)

movement_events = Table(
    "movement_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_time", DateTime(timezone=True), nullable=False),
    Column("camera_id", Text, nullable=False),
    Column("track_id", Integer, nullable=False),
    Column("employee_id", String(32)),
    Column("employee_name", Text, nullable=False, server_default="Chưa xác định"),
    Column("from_zone", Text),
    Column("to_zone", Text),
)
Index("idx_movement_time", movement_events.c.event_time.desc())

zone_visits = Table(
    "zone_visits", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("camera_id", Text, nullable=False),
    Column("track_id", Integer, nullable=False),
    Column("employee_id", String(32)),
    Column("employee_name", Text, nullable=False, server_default="Chưa xác định"),
    Column("zone_id", Text, nullable=False),
    Column("entered_at", DateTime(timezone=True), nullable=False),
    Column("exited_at", DateTime(timezone=True), nullable=False),
    Column("dwell_seconds", Float, nullable=False),
    CheckConstraint("dwell_seconds >= 0"),
)
Index("idx_visits_time", zone_visits.c.entered_at.desc())

activity_events = Table(
    "activity_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("employee_id", String(32)),
    Column("employee_name", Text, nullable=False, server_default="Chưa xác định"),
    Column("camera_id", Text, nullable=False),
    Column("track_id", Integer, nullable=False),
    Column("activity_type", String(32), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("ended_at", DateTime(timezone=True)),
    Column("duration_seconds", Float),
    Column("confidence", Float, nullable=False),
    Column("zone", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index("idx_activity_events_time", activity_events.c.started_at.desc())
Index("idx_activity_events_employee", activity_events.c.employee_id, activity_events.c.started_at.desc())

MAIN_TABLES = (employees, attendance, work_schedules, auth_settings, audit_logs, employee_accounts, login_limits)
SPATIAL_TABLES = (density_samples, movement_events, zone_visits, activity_events)

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2})?$")


def normalize_parameter(value: Any) -> Any:
    if isinstance(value, str):
        if _DATE.fullmatch(value):
            return date.fromisoformat(value)
        if _DATETIME.fullmatch(value):
            return datetime.fromisoformat(value)
    return value


def normalize_row(mapping: Any) -> dict[str, Any]:
    result = dict(mapping)
    for key, value in result.items():
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                value = value.astimezone(ZoneInfo("Asia/Ho_Chi_Minh"))
            result[key] = value.isoformat(timespec="seconds")
        elif isinstance(value, date):
            result[key] = value.isoformat()
    return result


class ResultAdapter:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.rowcount = result.rowcount
        self.lastrowid = None

    def fetchone(self) -> dict[str, Any] | None:
        row = self.result.mappings().fetchone()
        return normalize_row(row) if row is not None else None

    def fetchall(self) -> list[dict[str, Any]]:
        return [normalize_row(row) for row in self.result.mappings().fetchall()]


class ConnectionAdapter:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    @staticmethod
    def _statement(sql: str, size: int) -> str:
        parts = sql.split("?")
        if len(parts) - 1 != size:
            raise ValueError("SQL parameter count mismatch")
        return "".join(part + (f":p{i}" if i < size else "") for i, part in enumerate(parts))

    def execute(self, sql: str, params: Sequence[Any] = ()) -> ResultAdapter:
        named = self._statement(sql, len(params))
        bound = {f"p{i}": normalize_parameter(value) for i, value in enumerate(params)}
        return ResultAdapter(self.connection.execute(text(named), bound))

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> ResultAdapter | None:
        if not rows:
            return None
        named = self._statement(sql, len(rows[0]))
        bound = [
            {f"p{i}": normalize_parameter(value) for i, value in enumerate(row)}
            for row in rows
        ]
        return ResultAdapter(self.connection.execute(text(named), bound))
