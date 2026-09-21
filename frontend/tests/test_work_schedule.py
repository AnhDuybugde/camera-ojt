from datetime import datetime, timedelta
from pathlib import Path

from attendance.attendance_rules import scheduled_absent_ids
from attendance.attendance_service import AttendanceService
from database.db import Database


WORK_DAY = "2026-09-14"


def make_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "schedule.db")
    db.add_employee(
        {"employee_id": "NV001", "full_name": "Nguyen Van A", "department": "AI"}
    )
    return db


def test_case_1_on_creates_check_in(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    db.save_work_schedule("NV001", WORK_DAY, "ON")

    result = AttendanceService(db).record("NV001", datetime(2026, 9, 14, 8, 0))

    assert result.action == "CHECK_IN"
    assert len(db.list_attendance()) == 1


def test_case_2_on_does_not_create_duplicate(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    db.save_work_schedule("NV001", WORK_DAY, "ON")
    service = AttendanceService(db)
    first = datetime(2026, 9, 14, 8, 0)

    assert service.record("NV001", first).action == "CHECK_IN"
    assert service.record("NV001", first + timedelta(seconds=1)).action == "COOLDOWN"
    assert len(db.list_attendance()) == 1


def test_case_3_wfh_does_not_create_attendance(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    db.save_work_schedule("NV001", WORK_DAY, "WFH")

    result = AttendanceService(db).record("NV001", datetime(2026, 9, 14, 8, 0))

    assert result.action == "WFH"
    assert result.message == "Ca này WFH - Không chấm công"
    assert db.list_attendance() == []


def test_case_4_off_does_not_create_attendance(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    db.save_work_schedule("NV001", WORK_DAY, "OFF")

    result = AttendanceService(db).record("NV001", datetime(2026, 9, 14, 8, 0))

    assert result.action == "OFF"
    assert result.message == "Ca này OFF - Không chấm công"
    assert db.list_attendance() == []


def test_case_5_unscheduled_does_not_create_attendance(tmp_path: Path) -> None:
    db = make_db(tmp_path)

    result = AttendanceService(db).record("NV001", datetime(2026, 9, 14, 8, 0))

    assert result.action == "NO_SCHEDULE"
    assert result.message == "Chưa đăng ký lịch ca này - Không chấm công"
    assert db.list_attendance() == []


def test_case_6_on_without_check_in_is_absent() -> None:
    assert scheduled_absent_ids({"NV001"}, set()) == {"NV001"}


def test_case_7_wfh_without_check_in_is_not_absent() -> None:
    assert scheduled_absent_ids(set(), set()) == set()


def test_case_8_off_without_check_in_is_not_absent() -> None:
    assert scheduled_absent_ids(set(), set()) == set()


def test_case_9_updating_wfh_to_on_enables_attendance(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    service = AttendanceService(db)
    at = datetime(2026, 9, 14, 8, 0)
    db.save_work_schedule("NV001", WORK_DAY, "WFH")
    assert service.record("NV001", at).action == "WFH"

    db.save_work_schedule("NV001", WORK_DAY, "ON")
    result = service.record("NV001", at + timedelta(seconds=1))

    assert result.action == "CHECK_IN"
    assert len(db.list_attendance()) == 1


def test_case_10_saving_same_day_updates_without_duplicate(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    db.save_work_schedule("NV001", WORK_DAY, "WFH")
    db.save_work_schedule("NV001", WORK_DAY, "ON")

    rows = db.list_work_schedules(WORK_DAY, WORK_DAY)

    assert len(rows) == 1
    assert rows[0]["work_status"] == "ON"


def test_morning_and_afternoon_are_independent(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    db.save_work_schedule("NV001", WORK_DAY, "OFF", "MORNING")
    db.save_work_schedule("NV001", WORK_DAY, "ON", "AFTERNOON")
    service = AttendanceService(db)

    assert service.record("NV001", datetime(2026, 9, 14, 8, 0)).action == "OFF"
    assert service.record("NV001", datetime(2026, 9, 14, 14, 0)).action == "CHECK_IN"


def test_open_morning_attendance_can_check_out_during_off_afternoon(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    db.save_work_schedule("NV001", WORK_DAY, "ON", "MORNING")
    db.save_work_schedule("NV001", WORK_DAY, "OFF", "AFTERNOON")
    service = AttendanceService(db)

    assert service.record("NV001", datetime(2026, 9, 14, 8, 0)).action == "CHECK_IN"
    assert service.record("NV001", datetime(2026, 9, 14, 17, 30)).action == "CHECK_OUT"


def test_lunch_break_does_not_create_attendance(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    db.save_work_schedule("NV001", WORK_DAY, "ON", "AFTERNOON")

    result = AttendanceService(db).record("NV001", datetime(2026, 9, 14, 12, 30))

    assert result.action == "BREAK"
    assert db.list_attendance() == []


def test_weekend_schedule_never_creates_attendance(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    saturday = "2026-09-19"
    db.save_work_schedule("NV001", saturday, "ON", "MORNING")

    result = AttendanceService(db).record("NV001", datetime(2026, 9, 19, 8, 0))

    assert result.action == "NO_SCHEDULE"
    assert db.list_attendance() == []
