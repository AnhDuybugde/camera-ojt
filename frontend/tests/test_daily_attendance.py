from datetime import datetime, timedelta
from pathlib import Path

import pytest

from attendance.attendance_service import AttendanceService
from attendance.admin_service import AttendanceAdminService
from attendance.daily_service import DailyAttendanceService, daily_attendance_counts
from database.db import Database
from google_sheets.sync_service import SyncService
from scripts.backfill_scheduled_absences import backfill
from ui.history import _status_tags


DAY = "2026-09-22"


@pytest.fixture
def services(tmp_path: Path):
    db = Database(tmp_path / "daily.db")
    for code in ("ON", "WFH", "OFF", "NONE"):
        db.add_employee({"employee_id": code, "full_name": code})
    for code in ("ON", "WFH", "OFF"):
        db.save_work_schedule(code, DAY, code)
    return db, DailyAttendanceService(db), AttendanceService(db)


def test_waiting_is_idempotent_and_only_on(services):
    db, daily, _ = services
    assert daily.ensure_daily_attendance_records(datetime.fromisoformat(DAY).date()) == 1
    assert daily.ensure_daily_attendance_records(datetime.fromisoformat(DAY).date()) == 0
    rows = db.list_attendance()
    assert len(rows) == 1
    assert rows[0]["employee_id"] == "ON"
    assert rows[0]["status"] == "WAITING"
    assert rows[0]["check_in"] is None
    assert daily_attendance_counts(rows, {"ON"})["absent"] == 0


@pytest.mark.parametrize("clock,status", [("09:05", "ON_TIME"), ("09:20", "LATE")])
def test_recognition_updates_waiting_row(services, clock, status):
    db, daily, attendance = services
    daily.ensure_daily_attendance_records(datetime.fromisoformat(DAY).date())
    original_id = db.list_attendance()[0]["id"]
    assert attendance.record("ON", datetime.fromisoformat(f"{DAY}T{clock}:00")).action == "CHECK_IN"
    rows = db.list_attendance()
    assert len(rows) == 1 and rows[0]["id"] == original_id
    assert rows[0]["status"] == status
    assert rows[0]["presence_status"] == "PRESENT"


def test_absent_then_late_is_same_row_and_audited(services):
    db, daily, attendance = services
    day = datetime.fromisoformat(DAY).date()
    daily.ensure_daily_attendance_records(day)
    original_id = db.list_attendance()[0]["id"]
    assert daily.update_missing_checkins(day, datetime.fromisoformat(f"{DAY}T09:29:59")) == 0
    assert daily.update_missing_checkins(day, datetime.fromisoformat(f"{DAY}T09:30:00")) == 1
    assert daily.update_missing_checkins(day, datetime.fromisoformat(f"{DAY}T09:31:00")) == 0
    row = db.list_attendance()[0]
    assert row["status"] == "ABSENT" and row["check_in"] is None
    assert _status_tags(row) == ["Vắng mặt"]
    assert daily_attendance_counts([row], {"ON"})["absent"] == 1
    assert attendance.record("ON", datetime.fromisoformat(f"{DAY}T10:00:00")).action == "CHECK_IN"
    row = db.list_attendance()[0]
    assert row["id"] == original_id and row["status"] == "LATE"
    assert row["check_in"] == f"{DAY}T10:00:00"
    assert {r["action"] for r in db.list_audit_logs()} >= {"MARK_ABSENT", "LATE_CHECK_IN"}


def test_no_schedule_wfh_off_never_absent(services):
    db, daily, _ = services
    day = datetime.fromisoformat(DAY).date()
    daily.reconcile_today(datetime.fromisoformat(f"{DAY}T10:00:00"))
    assert [row["employee_id"] for row in db.list_attendance()] == ["ON"]


def test_temporary_checkout_return_and_timeout_preserve_late(services):
    db, daily, attendance = services
    morning = datetime.fromisoformat(f"{DAY}T09:20:00")
    assert attendance.record("ON", morning).action == "CHECK_IN"
    assert daily.mark_temporary_checkout("ON", morning + timedelta(minutes=40))
    row = db.list_attendance()[0]
    assert (row["status"], row["presence_status"]) == ("LATE", "TEMP_CHECKOUT")
    assert attendance.record("ON", morning + timedelta(minutes=60)).action == "RETURN"
    assert db.list_attendance()[0]["presence_status"] == "PRESENT"
    assert daily.mark_temporary_checkout("ON", morning + timedelta(minutes=70))
    assert daily.update_presence_timeout(morning + timedelta(minutes=99)) == 0
    assert daily.update_presence_timeout(morning + timedelta(minutes=100)) == 1
    row = db.list_attendance()[0]
    assert (row["status"], row["presence_status"]) == ("LATE", "ABSENT")
    assert row["temporary_checkout_at"]
    assert attendance.record("ON", morning + timedelta(minutes=101)).action == "RETURN"
    assert db.list_attendance()[0]["presence_status"] == "PRESENT"


def test_lunch_does_not_count_as_presence_timeout(services):
    db, daily, attendance = services
    assert attendance.record("ON", datetime.fromisoformat(f"{DAY}T09:05:00")).action == "CHECK_IN"
    assert daily.mark_temporary_checkout("ON", datetime.fromisoformat(f"{DAY}T11:55:00"))
    assert not daily.mark_temporary_checkout("ON", datetime.fromisoformat(f"{DAY}T12:05:00"))
    assert daily.update_presence_timeout(datetime.fromisoformat(f"{DAY}T14:20:00")) == 0
    assert daily.update_presence_timeout(datetime.fromisoformat(f"{DAY}T14:25:00")) == 1
    assert db.list_attendance()[0]["status"] == "ON_TIME"


def test_afternoon_only_absence_uses_shift_offset(tmp_path: Path):
    db = Database(tmp_path / "afternoon.db")
    db.add_employee({"employee_id": "PM", "full_name": "PM"})
    db.save_work_schedule("PM", DAY, "ON", "AFTERNOON")
    daily = DailyAttendanceService(db)
    day = datetime.fromisoformat(DAY).date()
    daily.ensure_daily_attendance_records(day)
    assert daily.update_missing_checkins(day, datetime.fromisoformat(f"{DAY}T09:30:00")) == 0
    assert daily.update_missing_checkins(day, datetime.fromisoformat(f"{DAY}T14:30:00")) == 1


def test_absent_row_reaches_reporting_mirror(services):
    db, daily, _ = services
    daily.reconcile_today(datetime.fromisoformat(f"{DAY}T09:30:00"))

    class FakeSheets:
        configured = True
        rows = []

        def upsert_attendance(self, row):
            self.rows.append(row.copy())

    client = FakeSheets()
    assert SyncService(db, client).sync_pending(actor_role="ADMIN").synced == 1
    assert client.rows[0]["status"] == "ABSENT"
    assert client.rows[0]["check_in"] is None


def test_historical_backfill_is_preview_only_without_explicit_apply(services):
    db, _, _ = services
    day = datetime(2026, 9, 14).date()
    db.save_work_schedule("ON", day.isoformat(), "ON")
    assert backfill(db, day, day) == (1, 0)
    assert db.list_attendance() == []
    assert backfill(db, day, day, apply=True) == (1, 1)
    assert backfill(db, day, day, apply=True) == (0, 0)
    assert db.list_attendance()[0]["status"] == "ABSENT"


def test_admin_can_correct_absent_to_late_with_audit(services):
    db, daily, _ = services
    daily.reconcile_today(datetime.fromisoformat(f"{DAY}T09:30:00"))
    attendance_id = db.list_attendance()[0]["id"]
    updated = AttendanceAdminService(db).update_attendance(
        attendance_id, actor_role="ADMIN", work_date=DAY,
        check_in=f"{DAY}T10:00:00", check_out=None,
        reason="Đã xác minh thời gian đến",
    )
    assert updated["status"] == "LATE"
    assert updated["presence_status"] == "PRESENT"
    assert any(row["action"] == "UPDATE_ATTENDANCE" for row in db.list_audit_logs())
