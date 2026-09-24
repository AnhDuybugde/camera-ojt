from datetime import datetime, timedelta
from pathlib import Path

from attendance.attendance_service import AttendanceService
from config import settings
from database.db import Database


def make_service(tmp_path: Path) -> tuple[Database, AttendanceService]:
    db = Database(tmp_path / "attendance.db")
    db.add_employee({"employee_id": "NV001", "full_name": "Nguyen Van A", "department": "AI"})
    db.save_work_schedule("NV001", "2026-09-14", "ON")
    return db, AttendanceService(db)


def test_duplicate_and_checkout_flow(tmp_path: Path) -> None:
    db, service = make_service(tmp_path)
    check_in = datetime(2026, 9, 14, 8, 0)
    assert service.record("NV001", check_in).action == "CHECK_IN"
    assert service.record("NV001", check_in + timedelta(seconds=10)).action == "COOLDOWN"
    wait_time = max(settings.attendance_cooldown + 1, settings.checkout_min_minutes * 60 - 1)
    assert service.record("NV001", check_in + timedelta(seconds=wait_time)).action == "WAITING"
    assert service.record("NV001", check_in + timedelta(minutes=settings.checkout_min_minutes, seconds=settings.attendance_cooldown + 2)).action == "CHECK_OUT"
    row = db.list_attendance()[0]
    assert row["check_in"] and row["check_out"]
    assert row["sync_status"] == "PENDING"


def test_unknown_employee_does_not_create_record(tmp_path: Path) -> None:
    db, service = make_service(tmp_path)
    assert service.record("MISSING").action == "ERROR"
    assert db.list_attendance() == []


def test_afternoon_checkin_uses_afternoon_late_cutoff(tmp_path: Path) -> None:
    db = Database(tmp_path / "attendance.db")
    db.add_employee({"employee_id": "NV001", "full_name": "Nguyen Van A"})
    day = "2026-09-14"
    db.save_work_schedules([
        ("NV001", day, "MORNING", "ON"),
        ("NV001", day, "AFTERNOON", "ON"),
    ])
    service = AttendanceService(db)

    assert service.record("NV001", datetime(2026, 9, 14, 14, 15)).record["status"] == "ON_TIME"


def test_afternoon_checkin_after_1415_is_late(tmp_path: Path) -> None:
    db = Database(tmp_path / "attendance.db")
    db.add_employee({"employee_id": "NV001", "full_name": "Nguyen Van A"})
    day = "2026-09-14"
    db.save_work_schedule("NV001", day, "ON", "AFTERNOON")
    service = AttendanceService(db)

    assert service.record("NV001", datetime(2026, 9, 14, 14, 16)).record["status"] == "LATE"
