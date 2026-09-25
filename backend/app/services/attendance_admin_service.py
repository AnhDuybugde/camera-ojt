"""Administrator-only attendance corrections with atomic audit logging."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import date, datetime

from backend.app.services.attendance_rules import check_in_status
from auth.permissions import require_permission
from config import settings
from backend.app.database.database import Database
from utils.helpers import iso_now


ATTENDANCE_STATUSES = frozenset({"ON_TIME", "LATE", "ABSENT"})
AUDITED_FIELDS = ("date", "check_in", "check_out", "status")


class AttendanceAdminService:
    def __init__(self, db: Database, on_change: Callable[[], None] | None = None) -> None:
        self.db = db
        self.on_change = on_change

    def update_attendance(
        self,
        attendance_id: int,
        *,
        actor_role: str,
        work_date: str,
        check_in: str | None,
        check_out: str | None,
        reason: str,
        status_override: str | None = None,
    ) -> dict:
        require_permission(actor_role, "attendance.update")
        reason = reason.strip()
        if not reason:
            raise ValueError("Bắt buộc nhập lý do chỉnh sửa.")
        normalized_date = date.fromisoformat(work_date).isoformat()
        normalized_check_in = self._normalize_timestamp(check_in, normalized_date)
        normalized_check_out = self._normalize_timestamp(check_out, normalized_date)
        if normalized_check_in and normalized_check_out:
            if datetime.fromisoformat(normalized_check_out) < datetime.fromisoformat(normalized_check_in):
                raise ValueError("Giờ check-out không được sớm hơn giờ check-in.")

        override = status_override.strip().upper() if status_override else None
        if override and override not in ATTENDANCE_STATUSES:
            raise ValueError("Trạng thái attendance không hợp lệ.")
        automatic_status = (
            check_in_status(datetime.fromisoformat(normalized_check_in), settings.late_threshold)
            if normalized_check_in else "ABSENT"
        )
        final_status = override or automatic_status
        timestamp = iso_now()

        with self.db.transaction() as conn:
            old_row = conn.execute(
                "SELECT * FROM attendance WHERE id=?", (attendance_id,)
            ).fetchone()
            if old_row is None:
                raise ValueError("Không tìm thấy bản ghi chấm công.")
            old_values = {field: old_row[field] for field in AUDITED_FIELDS}
            new_values = {
                "date": normalized_date,
                "check_in": normalized_check_in,
                "check_out": normalized_check_out,
                "status": final_status,
            }
            try:
                conn.execute(
                    """UPDATE attendance
                       SET date=?, check_in=?, check_out=?, status=?,
                           sync_status='PENDING', updated_at=?
                       WHERE id=?""",
                    (normalized_date, normalized_check_in, normalized_check_out,
                     final_status, timestamp, attendance_id),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Nhân viên đã có bản ghi chấm công trong ngày này.") from exc
            conn.execute(
                """INSERT INTO audit_logs
                   (timestamp, role, action, employee_id, old_values, new_values, reason)
                   VALUES (?, ?, 'UPDATE_ATTENDANCE', ?, ?, ?, ?)""",
                (
                    timestamp,
                    actor_role.upper(),
                    old_row["employee_id"],
                    json.dumps(old_values, ensure_ascii=False),
                    json.dumps(new_values, ensure_ascii=False),
                    reason,
                ),
            )

        if self.on_change:
            self.on_change()
        updated = self.db.get_attendance(attendance_id)
        if updated is None:
            raise RuntimeError("Không thể đọc lại bản ghi vừa cập nhật.")
        return updated

    @staticmethod
    def _normalize_timestamp(value: str | None, work_date: str) -> str | None:
        if not value:
            return None
        parsed = datetime.fromisoformat(value)
        return datetime.combine(date.fromisoformat(work_date), parsed.time()).isoformat(
            timespec="seconds"
        )
