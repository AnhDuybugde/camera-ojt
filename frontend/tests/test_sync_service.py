from datetime import datetime
from pathlib import Path

from attendance.attendance_service import AttendanceService
from database.db import Database
from google_sheets.sync_service import SyncService
from google_sheets.sheets_client import SheetsClient


class FakeSheetsClient:
    configured = True

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.rows = []

    def upsert_attendance(self, record: dict) -> None:
        if self.fail:
            raise ConnectionError("offline")
        self.rows.append(record)


def prepared_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "sync.db")
    db.add_employee({"employee_id": "NV001", "full_name": "Nguyen Van A"})
    db.save_work_schedule("NV001", "2026-09-14", "ON")
    AttendanceService(db).record("NV001", datetime(2026, 9, 14, 8, 0))
    return db


def test_successful_sync_marks_record(tmp_path: Path) -> None:
    db, client = prepared_db(tmp_path), FakeSheetsClient()
    result = SyncService(db, client).sync_pending(actor_role="ADMIN")
    assert (result.synced, result.failed) == (1, 0)
    assert db.list_attendance()[0]["sync_status"] == "SYNCED"


def test_failed_sync_remains_retryable(tmp_path: Path) -> None:
    db = prepared_db(tmp_path)
    result = SyncService(db, FakeSheetsClient(fail=True)).sync_pending(actor_role="ADMIN")
    assert (result.synced, result.failed) == (0, 1)
    assert db.list_attendance()[0]["sync_status"] == "ERROR"
    assert len(db.pending_attendance()) == 1


def test_absent_google_sheet_row_has_blank_times_and_vietnamese_status() -> None:
    class Worksheet:
        appended = None

        def get_all_values(self):
            return [["Employee ID", "Employee Name", "Department", "Date", "Check In", "Check Out", "Status"]]

        def append_row(self, values, value_input_option):
            self.appended = values

    worksheet = Worksheet()
    client = SheetsClient()
    client._worksheet = worksheet
    client.upsert_attendance({"employee_id": "NV001", "employee_name": "A", "department": "AI",
                              "date": "2026-09-22", "check_in": None, "check_out": None,
                              "status": "ABSENT"})
    assert worksheet.appended == ["NV001", "A", "AI", "2026-09-22", "", "", "Vắng mặt"]
