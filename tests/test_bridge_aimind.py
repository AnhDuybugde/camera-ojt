"""Test ai-mind-attendance bridge: queue ticks -> SQLite via record()."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "attendance"))

from camera_tracking.store.aimind_bridge import (
    AIMIND_TICK_KIND,
    build_checkin_payload,
    build_checkout_payload,
    drain_once,
    load_person_map,
)
from camera_tracking.store.queue import WriteQueue

MONDAY = "2026-09-21"  # a Monday: record() skips weekends without schedule
CHECKIN_AT = f"{MONDAY}T08:00:00+07:00"
CHECKOUT_AT = f"{MONDAY}T17:00:00+07:00"


def _aimind_db(tmp_path):
    from database.db import Database
    db = Database(tmp_path / "aim.db")
    db.add_employee({"employee_id": "NV001", "full_name": "Nguyen Van A",
                     "department": "AI"})
    db.save_work_schedule("NV001", MONDAY, "ON", "MORNING")
    db.save_work_schedule("NV001", MONDAY, "ON", "AFTERNOON")
    return db


def _service(db):
    from attendance.attendance_service import AttendanceService
    return AttendanceService(db)


def test_builders_shape() -> None:
    payload = build_checkin_payload(
        day=MONDAY, person_id="AnhDuy", display_name="Anh Duy",
        global_id=7, wall_iso=CHECKIN_AT, face_score=0.9)
    assert payload["tick"] == "CHECK_IN"
    assert payload["person_id"] == "AnhDuy" and payload["at"] == CHECKIN_AT
    payload = build_checkout_payload(
        day=MONDAY, person_id="AnhDuy", display_name="Anh Duy",
        global_id=7, wall_iso=CHECKOUT_AT)
    assert payload["tick"] == "CHECK_OUT"


def test_load_person_map_missing_is_empty(tmp_path) -> None:
    assert load_person_map(tmp_path / "nope.json") == {}


def test_drain_checkin_then_checkout(tmp_path) -> None:
    db = _aimind_db(tmp_path)
    service = _service(db)
    queue = WriteQueue(tmp_path / "q.db")
    queue.push(AIMIND_TICK_KIND, build_checkin_payload(
        day=MONDAY, person_id="AnhDuy", display_name="Anh Duy",
        global_id=7, wall_iso=CHECKIN_AT, face_score=0.9))
    stats = drain_once(queue, service, {"AnhDuy": "NV001"})
    assert stats["checked_in"] == 1 and len(queue) == 0
    row = db.fetch_one(
        "SELECT * FROM attendance WHERE employee_id=? AND date=?",
        ("NV001", MONDAY))
    assert row is not None and row["check_in"] is not None
    assert row["check_out"] is None

    queue.push(AIMIND_TICK_KIND, build_checkout_payload(
        day=MONDAY, person_id="AnhDuy", display_name="Anh Duy",
        global_id=7, wall_iso=CHECKOUT_AT))
    stats = drain_once(queue, service, {"AnhDuy": "NV001"})
    assert stats["checked_out"] == 1 and len(queue) == 0
    row = db.fetch_one(
        "SELECT * FROM attendance WHERE employee_id=? AND date=?",
        ("NV001", MONDAY))
    assert row["check_out"] is not None


def test_drain_skips_unmapped(tmp_path) -> None:
    db = _aimind_db(tmp_path)
    queue = WriteQueue(tmp_path / "q.db")
    queue.push(AIMIND_TICK_KIND, build_checkin_payload(
        day=MONDAY, person_id="Stranger", display_name="Stranger",
        global_id=9, wall_iso=CHECKIN_AT, face_score=0.9))
    stats = drain_once(queue, _service(db), {"AnhDuy": "NV001"})
    assert stats["skipped_unmapped"] == 1 and len(queue) == 0
    assert db.fetch_one(
        "SELECT * FROM attendance WHERE employee_id=? AND date=?",
        ("NV001", MONDAY)) is None


def test_drain_uses_matching_employee_id_without_map(tmp_path) -> None:
    db = _aimind_db(tmp_path)
    queue = WriteQueue(tmp_path / "q.db")
    queue.push(AIMIND_TICK_KIND, build_checkin_payload(
        day=MONDAY, person_id="NV001", display_name="Nguyen Van A",
        global_id=7, wall_iso=CHECKIN_AT, face_score=0.9))
    stats = drain_once(queue, _service(db), {})
    assert stats["checked_in"] == 1 and len(queue) == 0


def test_drain_skips_no_schedule_without_retry(tmp_path) -> None:
    from database.db import Database
    db = Database(tmp_path / "aim.db")
    db.add_employee({"employee_id": "NV009", "full_name": "No Schedule"})
    queue = WriteQueue(tmp_path / "q.db")
    queue.push(AIMIND_TICK_KIND, build_checkin_payload(
        day=MONDAY, person_id="AnhDuy", display_name="Anh Duy",
        global_id=7, wall_iso=CHECKIN_AT, face_score=0.9))
    stats = drain_once(queue, _service(db), {"AnhDuy": "NV009"})
    assert stats["skipped_rule"] == 1 and len(queue) == 0


def test_drain_leaves_row_on_db_error(tmp_path) -> None:
    queue = WriteQueue(tmp_path / "q.db")
    queue.push(AIMIND_TICK_KIND, build_checkin_payload(
        day=MONDAY, person_id="AnhDuy", display_name="Anh Duy",
        global_id=7, wall_iso=CHECKIN_AT, face_score=0.9))

    class _Boom:
        def record(self, *args, **kwargs):
            raise RuntimeError("db locked")

    stats = drain_once(queue, _Boom(), {"AnhDuy": "NV001"})
    assert stats["errors"] == 1 and len(queue) == 1


def test_other_kinds_untouched(tmp_path) -> None:
    db = _aimind_db(tmp_path)
    queue = WriteQueue(tmp_path / "q.db")
    queue.push("event", {"event": "CHECK_IN"})
    stats = drain_once(queue, _service(db), {"AnhDuy": "NV001"})
    assert stats["processed"] == 0 and len(queue) == 1


def test_aimind_rows_are_not_starved_by_other_queue_kinds(tmp_path) -> None:
    db = _aimind_db(tmp_path)
    queue = WriteQueue(tmp_path / "q.db")
    for index in range(110):
        queue.push("event", {"index": index})
    queue.push(AIMIND_TICK_KIND, build_checkin_payload(
        day=MONDAY, person_id="NV001", display_name="Nguyen Van A",
        global_id=7, wall_iso=CHECKIN_AT, face_score=0.9))
    stats = drain_once(queue, _service(db), {})
    assert stats["checked_in"] == 1
    assert len(queue) == 110
