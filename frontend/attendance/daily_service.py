"""Schedule-driven daily attendance and explicit workplace-presence transitions.

One attendance row represents a person/day. Camera recognition updates that row;
it does not decide whether the row exists. Presence changes require an explicit
exit event: a missed face frame is not evidence that somebody left the office.
"""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import date, datetime, time, timedelta

from config import settings
from database.db import Database
from attendance.attendance_rules import work_session_at

logger = logging.getLogger(__name__)


def _first_on_sessions(db: Database, work_day: date) -> dict[str, str]:
    rows = db.list_work_schedules(work_day.isoformat(), work_day.isoformat())
    first: dict[str, str] = {}
    for row in rows:
        if row["work_status"] == "ON":
            employee_id = row["employee_id"]
            session = row["work_session"]
            if employee_id not in first or session == "MORNING":
                first[employee_id] = session
    return first


def _shift_time(session: str, morning_time: time) -> time:
    if session == "MORNING":
        return morning_time
    offset = datetime.combine(date.min, morning_time) - datetime.combine(date.min, settings.work_start_time)
    return (datetime.combine(date.min, settings.afternoon_start_time) + offset).time()


def scheduled_check_in_status(session: str, now: datetime) -> str:
    return "ON_TIME" if now.time().replace(tzinfo=None) <= _shift_time(session, settings.late_threshold) else "LATE"


def daily_attendance_counts(rows: list[dict], scheduled_on_ids: set[str]) -> dict[str, int]:
    """Shared dashboard/report counts; WAITING is never counted as absent."""
    tracked = [row for row in rows if row["employee_id"] in scheduled_on_ids]
    return {
        "scheduled_on": len(scheduled_on_ids),
        "present": sum(bool(row["check_in"]) and row["status"] != "ABSENT" for row in tracked),
        "late": sum(row["status"] == "LATE" for row in tracked),
        "absent": sum(row["status"] == "ABSENT" for row in tracked),
        "waiting": sum(row["status"] == "WAITING" for row in tracked),
    }


def _audit(conn, at: str, action: str, employee_id: str, old: dict, new: dict, reason: str) -> None:
    conn.execute(
        """INSERT INTO audit_logs(timestamp,role,action,employee_id,old_values,new_values,reason)
           VALUES (?,'SYSTEM',?,?,?,?,?)""",
        (at, action, employee_id, json.dumps(old, ensure_ascii=False),
         json.dumps(new, ensure_ascii=False), reason),
    )


def _working_minutes(start: datetime, end: datetime) -> float:
    """Count only configured office hours, excluding lunch and after work."""
    total = 0.0
    day = start.date()
    while day <= end.date():
        for begin, finish in ((settings.work_start_time, settings.morning_end_time),
                              (settings.afternoon_start_time, settings.work_end_time)):
            left = max(start, datetime.combine(day, begin))
            right = min(end, datetime.combine(day, finish))
            if right > left:
                total += (right - left).total_seconds() / 60
        day += timedelta(days=1)
    return total


class DailyAttendanceService:
    def __init__(self, db: Database, on_change: Callable[[], None] | None = None) -> None:
        self.db = db
        self.on_change = on_change

    def _notify(self, changed: int) -> None:
        if changed and self.on_change:
            try:
                self.on_change()
            except Exception:
                logger.exception("Could not notify attendance change listener")

    def ensure_daily_attendance_records(self, work_day: date) -> int:
        """Idempotently create WAITING rows only for employees with an ON session."""
        scheduled = _first_on_sessions(self.db, work_day)
        if not scheduled:
            return 0
        employees = {row["employee_id"]: row for row in self.db.list_employees()}
        timestamp = datetime.now().isoformat(timespec="seconds")
        inserted = 0
        with self.db.transaction() as conn:
            for employee_id in scheduled:
                employee = employees.get(employee_id)
                if employee is None:
                    continue
                cursor = conn.execute(
                    """INSERT INTO attendance
                       (employee_id,employee_name,department,date,check_in,check_out,status,
                        presence_status,sync_status,created_at,updated_at)
                       VALUES (?,?,?,?,NULL,NULL,'WAITING','ABSENT','PENDING',?,?)
                       ON CONFLICT(employee_id,date) DO NOTHING""",
                    (employee_id, employee["full_name"], employee["department"],
                     work_day.isoformat(), timestamp, timestamp),
                )
                inserted += cursor.rowcount
        self._notify(inserted)
        return inserted

    def update_missing_checkins(self, work_day: date, now: datetime) -> int:
        """Mark only elapsed ON obligations ABSENT; no historical blind backfill."""
        if now.date() != work_day:
            raise ValueError("Automatic absence reconciliation is limited to today")
        scheduled = _first_on_sessions(self.db, work_day)
        changed = 0
        timestamp = now.isoformat(timespec="seconds")
        with self.db.transaction() as conn:
            for employee_id, session in scheduled.items():
                if now.time().replace(tzinfo=None) < _shift_time(session, settings.absent_after):
                    continue
                row = conn.execute(
                    "SELECT id,status,check_in FROM attendance WHERE employee_id=? AND date=?",
                    (employee_id, work_day.isoformat()),
                ).fetchone()
                if not row or row["check_in"] or row["status"] != "WAITING":
                    continue
                result = conn.execute(
                    """UPDATE attendance SET status='ABSENT',sync_status='PENDING',updated_at=?
                       WHERE id=? AND status='WAITING' AND check_in IS NULL""",
                    (timestamp, row["id"]),
                )
                if result.rowcount:
                    _audit(conn, timestamp, "MARK_ABSENT", employee_id,
                           {"status": "WAITING"}, {"status": "ABSENT"},
                           "Không ghi nhận check-in trước mốc vắng mặt")
                    changed += 1
        self._notify(changed)
        return changed

    def mark_temporary_checkout(self, employee_id: str, now: datetime) -> bool:
        """Consume an explicit exit event; never infer departure from face absence."""
        current = now.time().replace(tzinfo=None)
        if not (settings.work_start_time <= current < settings.morning_end_time
                or settings.afternoon_start_time <= current < settings.work_end_time):
            return False
        session = work_session_at(now, settings.morning_end_time, settings.afternoon_start_time)
        schedule = self.db.get_work_schedule(employee_id, now.date().isoformat(), session) if session else None
        if not schedule or schedule["work_status"] != "ON":
            return False
        timestamp = now.isoformat(timespec="seconds")
        with self.db.transaction() as conn:
            row = conn.execute("SELECT * FROM attendance WHERE employee_id=? AND date=?",
                               (employee_id, now.date().isoformat())).fetchone()
            if not row or not row["check_in"] or row["check_out"] or row["presence_status"] != "PRESENT":
                return False
            result = conn.execute(
                """UPDATE attendance SET presence_status='TEMP_CHECKOUT',temporary_checkout_at=?,
                   sync_status='PENDING',updated_at=? WHERE id=? AND presence_status='PRESENT'""",
                (timestamp, timestamp, row["id"]),
            )
            if result.rowcount:
                _audit(conn, timestamp, "TEMP_CHECKOUT", employee_id,
                       {"presence_status": "PRESENT"},
                       {"presence_status": "TEMP_CHECKOUT", "temporary_checkout_at": timestamp},
                       "Ghi nhận sự kiện rời văn phòng")
        self._notify(result.rowcount)
        return bool(result.rowcount)

    def record_return(self, employee_id: str, now: datetime) -> bool:
        timestamp = now.isoformat(timespec="seconds")
        with self.db.transaction() as conn:
            row = conn.execute("SELECT * FROM attendance WHERE employee_id=? AND date=?",
                               (employee_id, now.date().isoformat())).fetchone()
            if not row or not row["check_in"] or row["check_out"] or row["presence_status"] not in {"TEMP_CHECKOUT", "ABSENT"}:
                return False
            old = {"presence_status": row["presence_status"],
                   "temporary_checkout_at": row["temporary_checkout_at"]}
            result = conn.execute(
                """UPDATE attendance SET presence_status='PRESENT',temporary_checkout_at=NULL,
                   sync_status='PENDING',updated_at=? WHERE id=? AND presence_status=?""",
                (timestamp, row["id"], row["presence_status"]),
            )
            if result.rowcount:
                _audit(conn, timestamp, "RETURN_TO_WORK", employee_id, old,
                       {"presence_status": "PRESENT", "returned_at": timestamp},
                       "Nhân viên trở lại sau check-out tạm thời")
        self._notify(result.rowcount)
        return bool(result.rowcount)

    def update_presence_timeout(self, now: datetime) -> int:
        timestamp = now.isoformat(timespec="seconds")
        changed = 0
        with self.db.transaction() as conn:
            rows = conn.execute(
                """SELECT * FROM attendance WHERE date=? AND presence_status='TEMP_CHECKOUT'
                   AND temporary_checkout_at IS NOT NULL AND check_out IS NULL""",
                (now.date().isoformat(),),
            ).fetchall()
            for row in rows:
                started = datetime.fromisoformat(row["temporary_checkout_at"])
                if _working_minutes(started, now) < settings.presence_timeout_minutes:
                    continue
                result = conn.execute(
                    """UPDATE attendance SET presence_status='ABSENT',sync_status='PENDING',
                       updated_at=? WHERE id=? AND presence_status='TEMP_CHECKOUT'""",
                    (timestamp, row["id"]),
                )
                if result.rowcount:
                    _audit(conn, timestamp, "PRESENCE_TIMEOUT", row["employee_id"],
                           {"presence_status": "TEMP_CHECKOUT", "temporary_checkout_at": row["temporary_checkout_at"]},
                           {"presence_status": "ABSENT"}, "Quá thời gian rời văn phòng cho phép")
                    changed += 1
        self._notify(changed)
        return changed

    def reconcile_today(self, now: datetime | None = None) -> tuple[int, int, int]:
        now = now or datetime.now()
        created = self.ensure_daily_attendance_records(now.date())
        absent = self.update_missing_checkins(now.date(), now)
        presence = self.update_presence_timeout(now)
        return created, absent, presence


class DailyAttendanceWorker:
    """Periodic non-blocking reconciliation while the application is running."""
    def __init__(self, service: DailyAttendanceService, interval_seconds: int = 30) -> None:
        self.service = service
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "DailyAttendanceWorker":
        if not self._thread or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True, name="daily-attendance")
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.service.reconcile_today()
            except Exception:
                logger.exception("Daily attendance reconciliation failed")
            self._stop.wait(self.interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
