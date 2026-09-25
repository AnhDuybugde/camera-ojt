"""Atomic check-in/check-out service with duplicate protection."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from collections.abc import Callable

from backend.app.services.attendance_rules import can_check_out, check_in_status, work_session_at
from config import settings
from backend.app.database.database import Database

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AttendanceResult:
    action: str
    message: str
    record: dict | None = None


class AttendanceService:
    def __init__(self, db: Database, on_change: Callable[[], None] | None = None) -> None:
        self.db = db
        self.on_change = on_change
        self._last_seen: dict[str, datetime] = {}
        self._lock = threading.RLock()

    def record_direction(self, employee_id: str, now: datetime, *, direction: str,
                         event_id: str, camera_id: str = "", session_id: str = "",
                         evidence: dict | None = None) -> AttendanceResult:
        """Commit a directional observation, daily projection and outbox atomically."""
        from zoneinfo import ZoneInfo
        from camera_tracking.store.event_log import append_event

        if direction not in {"CHECK_IN", "CHECK_OUT"} or now.tzinfo is None:
            raise ValueError("A directional event and timezone-aware timestamp are required")
        now = now.astimezone(ZoneInfo("Asia/Ho_Chi_Minh"))
        employee = self.db.get_employee(employee_id)
        if employee is None:
            return AttendanceResult("ERROR", "Employee no longer exists.")
        day, timestamp = now.date().isoformat(), now.isoformat(timespec="seconds")
        with self._lock, self.db.transaction() as conn:
            _, inserted = append_event(conn, event_id=event_id, employee_id=employee_id,
                kind=direction, observed_at=now, camera_id=camera_id,
                session_id=session_id, payload=evidence)
            if not inserted:
                return AttendanceResult("DUPLICATE", "Event already recorded.")
            row = conn.execute("SELECT * FROM attendance WHERE employee_id=? AND date=?",
                               (employee_id, day)).fetchone()
            action = direction
            if direction == "CHECK_OUT":
                if row is None or not row["check_in"] or timestamp < row["check_in"]:
                    conn.execute("UPDATE camera_events SET needs_review=1 WHERE event_id=?", (event_id,))
                    return AttendanceResult("REVIEW", "Exit has no preceding check-in.")
                # Every genuine exit is retained; the daily projection keeps the last.
                if not row["check_out"] or timestamp > row["check_out"]:
                    conn.execute("UPDATE attendance SET check_out=?, updated_at=?, sync_status='PENDING' WHERE id=?",
                                 (timestamp, timestamp, row["id"]))
            elif row is None:
                session = work_session_at(now, settings.morning_end_time, settings.afternoon_start_time)
                schedule = self.db.get_work_schedule(employee_id, day, session) if session else None
                rejection = ("NO_SCHEDULE" if now.weekday() >= 5 else
                             "BREAK" if session is None else
                             "NO_SCHEDULE" if schedule is None else
                             schedule["work_status"] if schedule["work_status"] != "ON" else None)
                if rejection:
                    return AttendanceResult(rejection, "Observation retained; schedule excludes attendance.")
                conn.execute("""INSERT INTO attendance
                    (employee_id, employee_name, department, date, check_in, status,
                     sync_status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)""",
                    (employee_id, employee["full_name"], employee["department"], day,
                     timestamp, check_in_status(now, settings.late_threshold), timestamp, timestamp))
            else:
                action = "RETURN"
        if self.on_change:
            try:
                self.on_change()
            except Exception:
                logger.exception("Could not notify attendance change listener")
        return AttendanceResult(action, "Directional event recorded.", self.db.fetch_one(
            "SELECT * FROM attendance WHERE employee_id=? AND date=?", (employee_id, day)))

    def record(self, employee_id: str, now: datetime | None = None) -> AttendanceResult:
        now = now or datetime.now()
        with self._lock:
            employee = self.db.get_employee(employee_id)
            if not employee:
                return AttendanceResult("ERROR", "Employee no longer exists.")
            day, timestamp = now.date().isoformat(), now.isoformat(timespec="seconds")
            existing = self.db.fetch_one(
                "SELECT * FROM attendance WHERE employee_id=? AND date=?",
                (employee_id, day),
            )
            # An open daily attendance record must remain eligible for CHECK-OUT.
            # Otherwise, validate the session active at the recognition time.
            if not (existing and existing["check_in"] and not existing["check_out"]):
                if now.weekday() >= 5:
                    return AttendanceResult(
                        "NO_SCHEDULE", "Không áp dụng lịch Thứ Bảy/Chủ Nhật - Không chấm công"
                    )
                work_session = work_session_at(
                    now, settings.morning_end_time, settings.afternoon_start_time
                )
                if work_session is None:
                    return AttendanceResult(
                        "BREAK", "Đang trong giờ nghỉ trưa - Không chấm công"
                    )
                schedule = self.db.get_work_schedule(employee_id, day, work_session)
                if schedule is None:
                    return AttendanceResult(
                        "NO_SCHEDULE", "Chưa đăng ký lịch ca này - Không chấm công"
                    )
                if schedule["work_status"] == "WFH":
                    return AttendanceResult("WFH", "Ca này WFH - Không chấm công")
                if schedule["work_status"] == "OFF":
                    return AttendanceResult("OFF", "Ca này OFF - Không chấm công")

            previous = self._last_seen.get(employee_id)
            if previous and (now - previous).total_seconds() < settings.attendance_cooldown:
                return AttendanceResult("COOLDOWN", "Already recorded recently.")
            with self.db.transaction() as conn:
                row = conn.execute(
                    "SELECT * FROM attendance WHERE employee_id=? AND date=?", (employee_id, day)
                ).fetchone()
                if row is None:
                    status = check_in_status(now, settings.late_threshold)
                    conn.execute(
                        """INSERT INTO attendance
                           (employee_id, employee_name, department, date, check_in, check_out,
                            status, sync_status, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, NULL, ?, 'PENDING', ?, ?)""",
                        (employee_id, employee["full_name"], employee["department"], day,
                         timestamp, status, timestamp, timestamp),
                    )
                    action = "CHECK_IN"
                    logger.info("Check-in: %s (%s)", employee["full_name"], employee_id)
                elif row["check_out"]:
                    action = "COMPLETE"
                elif not can_check_out(row["check_in"], now, settings.checkout_min_minutes):
                    action = "WAITING"
                else:
                    conn.execute(
                        """UPDATE attendance SET check_out=?, sync_status='PENDING', updated_at=?
                           WHERE id=?""", (timestamp, timestamp, row["id"])
                    )
                    action = "CHECK_OUT"
                    logger.info("Check-out: %s (%s)", employee["full_name"], employee_id)
            self._last_seen[employee_id] = now
            if action in {"CHECK_IN", "CHECK_OUT"} and self.on_change:
                try:
                    self.on_change()
                except Exception:
                    logger.exception("Could not notify attendance change listener")
            record = self.db.fetch_one(
                "SELECT * FROM attendance WHERE employee_id=? AND date=?", (employee_id, day)
            )
            messages = {
                "CHECK_IN": "CHECK-IN SUCCESS", "CHECK_OUT": "CHECK-OUT SUCCESS",
                "WAITING": "Checked in; checkout is not due yet.", "COMPLETE": "Attendance already complete.",
            }
            return AttendanceResult(action, messages[action], record)
