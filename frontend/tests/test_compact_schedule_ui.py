"""Behavior checks for the two role-specific views of the compact schedule."""
from datetime import date, datetime, timedelta

from streamlit.testing.v1 import AppTest

from attendance.schedule_policy import get_next_week_range
from database.db import Database


def _page(path: str, role: str) -> AppTest:
    source = '''
import streamlit as st
from database.db import Database
from ui.work_schedule import render
db = Database(PATH)
if not db.get_employee("NV001"):
    db.add_employee({"employee_id": "NV001", "full_name": "Nguyễn An", "department": "AI"})
    db.add_employee({"employee_id": "NV002", "full_name": "Trần Bình", "department": "Dev"})
st.session_state.role = ROLE
st.session_state.employee_id = "NV001" if ROLE == "EMPLOYEE" else None
render(db, ROLE)
'''.replace("PATH", repr(path)).replace("ROLE", repr(role))
    return AppTest.from_string(source, default_timeout=15).run()


def test_admin_bulk_applies_only_selected_employees(tmp_path):
    path = str(tmp_path / "schedule-admin.db")
    at = _page(path, "ADMIN")
    assert not at.exception
    week = date.fromisoformat(at.session_state.schedule_week_start.isoformat())
    db = Database(path)
    db.save_work_schedule("NV001", week.isoformat(), "WFH", "MORNING")
    db.save_work_schedule("NV002", week.isoformat(), "OFF", "MORNING")
    at.run()
    assert at.button(key="sg_batch_apply").disabled
    at.checkbox(key="sg_select_NV001").check().run()
    assert not at.exception
    assert not at.button(key="sg_batch_apply").disabled
    at.button(key="sg_batch_apply").click().run()
    assert not at.exception
    rows = db.list_work_schedules(week.isoformat(), (week + timedelta(days=6)).isoformat())
    assert len([row for row in rows if row["employee_id"] == "NV001" and row["work_status"] == "ON"]) == 10
    assert [(row["work_session"], row["work_status"]) for row in rows if row["employee_id"] == "NV002"] == [("MORNING", "OFF")]
    assert at.session_state.sg_select_NV001 is False


def test_employee_day_editor_saves_only_own_day_and_can_cancel(tmp_path):
    path = str(tmp_path / "schedule-employee.db")
    at = _page(path, "EMPLOYEE")
    assert not at.exception
    week, _ = get_next_week_range(datetime.now())
    db = Database(path)
    db.save_work_schedule("NV001", week.isoformat(), "ON", "MORNING")
    db.save_work_schedule("NV001", week.isoformat(), "WFH", "AFTERNOON")
    db.save_work_schedule("NV001", (week + timedelta(days=1)).isoformat(), "WFH", "MORNING")
    db.save_work_schedule("NV002", week.isoformat(), "OFF", "MORNING")
    at.run()
    markup = " ".join(x.value for x in at.markdown)
    assert "Nguyễn An" in markup and "Trần Bình" not in markup
    assert not at.checkbox and not at.text_input
    assert not at.exception
    token = f"EMPLOYEE_NV001_{week:%Y%m%d}"
    at.session_state[f"sg_edit_{token}"] = True
    at.run()
    assert not at.exception
    at.button(key=f"sg_quick_OFF_{token}").click().run()
    # AppTest does not transmit popover open state on an inner-button rerun.
    at.session_state[f"sg_edit_{token}"] = True
    at.run()
    at.button(key=f"sg_cancel_{token}").click().run()
    rows = db.list_work_schedules(week.isoformat(), week.isoformat(), employee_id="NV001")
    assert {row["work_session"]: row["work_status"] for row in rows} == {"MORNING": "ON", "AFTERNOON": "WFH"}

    at.session_state[f"sg_edit_{token}"] = True
    at.run()
    at.button(key=f"sg_quick_OFF_{token}").click().run()
    at.session_state[f"sg_edit_{token}"] = True
    at.run()
    at.button(key=f"sg_save_{token}").click().run()
    assert not at.exception
    rows = db.list_work_schedules(week.isoformat(), (week + timedelta(days=1)).isoformat())
    assert {row["work_session"]: row["work_status"] for row in rows
            if row["employee_id"] == "NV001" and row["work_date"] == week.isoformat()} == {"MORNING": "OFF", "AFTERNOON": "OFF"}
    assert [(row["employee_id"], row["work_date"], row["work_status"]) for row in rows
            if row["work_date"] != week.isoformat() or row["employee_id"] != "NV001"] == [
                ("NV001", (week + timedelta(days=1)).isoformat(), "WFH"),
                ("NV002", week.isoformat(), "OFF")]


def test_calendar_moves_to_selected_workweek_and_admin_filters_registration_status(tmp_path):
    path = str(tmp_path / "schedule-filter.db")
    at = _page(path, "ADMIN")
    db = Database(path)
    db.add_employee({"employee_id": "NV003", "full_name": "Lê Cường", "department": "AI"})
    week = at.session_state.schedule_week_start
    complete_entries = [
        ("NV001", (week + timedelta(days=offset)).isoformat(), session, "ON")
        for offset in range(5) for session in ("MORNING", "AFTERNOON")
    ]
    db.save_work_schedules(complete_entries)
    db.save_work_schedule("NV002", week.isoformat(), "WFH", "MORNING")
    at.run()
    assert not at.exception
    registration_filter = next(
        item for item in at.get("button_group") if item.key == "sg_registration_filter"
    )
    assert registration_filter.options == [
        "Tất cả (3)", "Đã hoàn thành (1)", "Chưa hoàn thành (1)", "Chưa đăng ký (1)"
    ]
    registration_filter.select("COMPLETE").run()
    markup = " ".join(item.value for item in at.markdown)
    assert "Nguyễn An" in markup
    assert "Trần Bình" not in markup and "Lê Cường" not in markup
    assert len(at.get("popover")) == 5

    next(item for item in at.get("button_group") if item.key == "sg_registration_filter").select("ALL").run()
    selected_day = week + timedelta(days=16)
    at.date_input(key="sg_calendar_date").set_value(selected_day).run()
    assert not at.exception
    assert at.session_state.schedule_week_start == selected_day - timedelta(days=selected_day.weekday())
    assert len(at.get("popover")) == 15
