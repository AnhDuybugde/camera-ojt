import json
from pathlib import Path

import pytest

from attendance.admin_service import AttendanceAdminService
from auth.permissions import PermissionDenied
from database.db import Database


def _record(db: Database) -> int:
    db.add_employee({"employee_id": "NV001", "full_name": "Nguyen Van A"})
    return db.execute(
        """INSERT INTO attendance
           (employee_id, employee_name, department, date, check_in, check_out,
            status, sync_status, created_at, updated_at)
           VALUES ('NV001', 'Nguyen Van A', '', '2026-09-14',
                   '2026-09-14T09:22:00', NULL, 'LATE', 'SYNCED',
                   '2026-09-14T09:22:00', '2026-09-14T09:22:00')"""
    )


def test_admin_edit_recalculates_status_and_writes_audit(tmp_path: Path) -> None:
    db = Database(tmp_path / "admin.db")
    attendance_id = _record(db)
    updated = AttendanceAdminService(db).update_attendance(
        attendance_id, actor_role="ADMIN", work_date="2026-09-14",
        check_in="2026-09-14T09:05:00", check_out="2026-09-14T17:30:00",
        reason="Máy chấm công ghi nhận trễ",
    )
    assert updated["status"] == "ON_TIME"
    assert updated["sync_status"] == "PENDING"
    audit = db.list_audit_logs()[0]
    assert audit["employee_id"] == "NV001"
    assert audit["reason"] == "Máy chấm công ghi nhận trễ"
    assert json.loads(audit["old_values"])["status"] == "LATE"
    assert json.loads(audit["new_values"])["status"] == "ON_TIME"


def test_admin_override_requires_reason_and_employee_is_denied(tmp_path: Path) -> None:
    db = Database(tmp_path / "admin.db")
    attendance_id = _record(db)
    service = AttendanceAdminService(db)
    kwargs = dict(
        work_date="2026-09-14", check_in="2026-09-14T09:05:00",
        check_out=None, status_override="ABSENT",
    )
    with pytest.raises(ValueError):
        service.update_attendance(attendance_id, actor_role="ADMIN", reason="", **kwargs)
    with pytest.raises(PermissionDenied):
        service.update_attendance(
            attendance_id, actor_role="EMPLOYEE", reason="Không có quyền", **kwargs
        )
    updated = service.update_attendance(
        attendance_id, actor_role="ADMIN", reason="Nghỉ không phép", **kwargs
    )
    assert updated["status"] == "ABSENT"
