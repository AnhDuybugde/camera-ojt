"""Optional real-PostgreSQL regression test in an isolated temporary schema."""
from datetime import datetime
import os
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from attendance.admin_service import AttendanceAdminService
from attendance.attendance_service import AttendanceService
from attendance.daily_service import DailyAttendanceService
from auth.service import AuthService
from database.db import Database
from google_sheets.sync_service import SyncService
from spatial.repository import SpatialRepository


@pytest.fixture(scope="module")
def postgres_url():
    base = os.getenv("TEST_DATABASE_URL", "")
    if not base.startswith("postgresql+psycopg://"):
        pytest.skip("TEST_DATABASE_URL is not configured")
    schema = "ai_mind_test_" + uuid4().hex[:12]
    engine = create_engine(base)
    with engine.begin() as conn:
        conn.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
    try:
        separator = "&" if "?" in base else "?"
        yield base + separator + "options=" + quote(f"-csearch_path={schema}", safe="")
    finally:
        with engine.begin() as conn:
            conn.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        engine.dispose()


def test_auth_face_attendance_audit_sheets_and_spatial(postgres_url, tmp_path: Path):
    db = Database(database_url=postgres_url)
    db.add_employee({"employee_id": "PGTEST", "full_name": "Postgres Test", "department": "AI"})
    auth = AuthService(db)
    assert auth.login("EMPLOYEE", "123", "PGTEST") is not None
    assert auth.login("ADMIN", "456") is not None

    db.save_embedding("PGTEST", b"\x01\x02\x03\x04", 1)
    assert db.list_embeddings()[0]["face_embedding"] == b"\x01\x02\x03\x04"
    db.save_work_schedule("PGTEST", "2026-09-28", "ON")
    service = AttendanceService(db)
    assert service.record("PGTEST", datetime(2026, 9, 28, 8, 0)).action == "CHECK_IN"
    assert service.record("PGTEST", datetime(2026, 9, 28, 17, 30)).action == "CHECK_OUT"
    assert len(db.list_attendance()) == 1

    attendance_id = db.list_attendance()[0]["id"]
    AttendanceAdminService(db).update_attendance(
        attendance_id, actor_role="ADMIN", work_date="2026-09-28",
        check_in="2026-09-28T08:05:00", check_out="2026-09-28T17:30:00",
        reason="PostgreSQL integration test",
    )
    assert any(row["action"] == "UPDATE_ATTENDANCE" for row in db.list_audit_logs())

    class FakeSheets:
        configured = True
        fail = True

        def upsert_attendance(self, record):
            if self.fail:
                raise ConnectionError("test offline")

    client = FakeSheets()
    sync = SyncService(db, client)
    assert sync.sync_pending(actor_role="ADMIN").failed == 1
    assert db.list_attendance()[0]["sync_status"] == "ERROR"
    client.fail = False
    assert sync.sync_pending(actor_role="ADMIN").synced == 1
    assert db.list_attendance()[0]["sync_status"] == "SYNCED"

    spatial = SpatialRepository(tmp_path / "unused.db", database_url=postgres_url)
    spatial.record_density("A", {"gate": 2}, datetime.now().astimezone())
    assert len(spatial.density_rows(datetime(2020, 1, 1).astimezone())) == 1


def test_postgres_waiting_absent_late_transition(postgres_url):
    db = Database(database_url=postgres_url)
    db.add_employee({"employee_id": "PGABS", "full_name": "Postgres Absence"})
    db.save_work_schedule("PGABS", "2026-09-22", "ON")
    daily = DailyAttendanceService(db)
    day = datetime(2026, 9, 22).date()
    assert daily.ensure_daily_attendance_records(day) == 1
    assert daily.ensure_daily_attendance_records(day) == 0
    assert daily.update_missing_checkins(day, datetime(2026, 9, 22, 9, 30)) == 1
    assert AttendanceService(db).record("PGABS", datetime(2026, 9, 22, 10)).action == "CHECK_IN"
    rows = db.list_attendance(employee_id="PGABS")
    assert len(rows) == 1 and rows[0]["status"] == "LATE"
