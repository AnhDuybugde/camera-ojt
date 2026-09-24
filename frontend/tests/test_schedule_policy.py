from datetime import date, datetime, timedelta

import pytest

from attendance.schedule_policy import (
    can_employee_edit_schedule, get_next_week_range, get_schedule_deadline,
    get_week_schedule_completion,
)
from database.db import Database


MONDAY = datetime(2026, 9, 21, 9, 0)
FRIDAY = datetime(2026, 9, 25, 23, 59, 59)
SATURDAY = datetime(2026, 9, 26, 0, 0)
NEXT_MONDAY = date(2026, 9, 28)


def test_employee_window_monday_friday_saturday():
    assert get_next_week_range(MONDAY) == (NEXT_MONDAY, date(2026, 10, 2))
    assert get_schedule_deadline(MONDAY) == FRIDAY
    assert can_employee_edit_schedule(NEXT_MONDAY, MONDAY)
    assert can_employee_edit_schedule("2026-10-02", FRIDAY)
    assert not can_employee_edit_schedule("2026-10-03", FRIDAY)
    assert not can_employee_edit_schedule(NEXT_MONDAY, SATURDAY)
    assert not can_employee_edit_schedule("2026-09-21", MONDAY)
    assert not can_employee_edit_schedule("2026-10-05", MONDAY)


def test_completion_requires_both_sessions_on_all_five_workdays():
    rows = []
    assert get_week_schedule_completion(rows, "NV001", NEXT_MONDAY).status == "UNREGISTERED"
    for offset in range(4):
        for session in ("MORNING", "AFTERNOON"):
            rows.append({"employee_id": "NV001", "work_date": (NEXT_MONDAY + timedelta(days=offset)).isoformat(),
                         "work_session": session, "work_status": "ON"})
    assert get_week_schedule_completion(rows, "NV001", NEXT_MONDAY).registered_days == 4
    assert get_week_schedule_completion(rows, "NV001", NEXT_MONDAY).status == "INCOMPLETE"
    rows.append({"employee_id": "NV001", "work_date": "2026-10-02", "work_session": "MORNING", "work_status": "OFF"})
    assert get_week_schedule_completion(rows, "NV001", NEXT_MONDAY).status == "INCOMPLETE"
    rows.append({"employee_id": "NV001", "work_date": "2026-10-02", "work_session": "AFTERNOON", "work_status": "OFF"})
    assert get_week_schedule_completion(rows, "NV001", NEXT_MONDAY).status == "COMPLETE"
    rows.extend([
        {"employee_id": "NV001", "work_date": "2026-10-03", "work_session": session, "work_status": "ON"}
        for session in ("MORNING", "AFTERNOON")
    ])
    assert get_week_schedule_completion(rows, "NV001", NEXT_MONDAY).registered_days == 5


def test_one_selected_session_is_incomplete_not_unregistered():
    rows = [{"employee_id": "NV001", "work_date": "2026-09-28", "work_session": "MORNING", "work_status": "ON"}]
    completion = get_week_schedule_completion(rows, "NV001", NEXT_MONDAY)
    assert completion.registered_days == 0
    assert completion.status == "INCOMPLETE"


def test_repository_rechecks_deadline_and_owner(tmp_path):
    db = Database(tmp_path / "policy.db")
    db.add_employee({"employee_id": "NV001", "full_name": "Employee"})
    entry = [("NV001", "2026-09-28", "MORNING", "ON")]
    db.save_work_schedules(entry, actor_role="EMPLOYEE", actor_employee_id="NV001", now=MONDAY)
    db.save_work_schedules(entry, actor_role="EMPLOYEE", actor_employee_id="NV001", now=FRIDAY)
    with pytest.raises(PermissionError):
        db.save_work_schedules(entry, actor_role="EMPLOYEE", actor_employee_id="NV001", now=SATURDAY)
    with pytest.raises(PermissionError):
        db.save_work_schedules([("NV001", "2026-09-21", "MORNING", "ON")], actor_role="EMPLOYEE", actor_employee_id="NV001", now=MONDAY)
    db.save_work_schedules(entry, actor_role="ADMIN", now=SATURDAY)
