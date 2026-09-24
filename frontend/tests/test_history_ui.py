from datetime import date, time
import json

from streamlit.testing.v1 import AppTest
from database.db import Database


def _app(tmp_path):
    path = tmp_path / "history-ui.db"
    db = Database(path)
    day = date.today().isoformat()
    for eid, name, department in [("NV001", "Nguyễn Văn Tên Rất Dài Cần Xuống Dòng", "AI"), ("NV002", "Trần Bình", "Dev")]:
        db.add_employee({"employee_id": eid, "full_name": name, "department": department})
        db.execute("""INSERT INTO attendance
            (employee_id, employee_name, department, date, check_in, status, sync_status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'LATE', 'PENDING', ?, ?)""",
            (eid, name, department, day, day + "T09:22:00", day + "T09:22:00", day + "T09:22:00"))
    source = f'''
from database.db import Database
from attendance.admin_service import AttendanceAdminService
from ui.history import render
db = Database({str(path)!r})
render(db, "ADMIN", AttendanceAdminService(db))
'''
    return AppTest.from_string(source, default_timeout=15).run(), db


def test_history_tabs_filters_and_csv(tmp_path):
    at, db = _app(tmp_path)
    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["Lịch sử chấm công", "Chỉnh sửa chấm công"]
    assert not at.tabs[0].text_area
    assert len(at.tabs[1].text_area) == 1
    assert len(at.get("download_button")) == 1
    html = " ".join(x.value for x in at.tabs[0].markdown)
    assert "Nguyễn Văn Tên Rất Dài Cần Xuống Dòng" in html
    assert "<th" in html and "Thao tác" not in html
    next(x for x in at.selectbox if x.label == "Phòng ban").select("Dev").run()
    assert not at.exception
    html = " ".join(x.value for x in at.tabs[0].markdown)
    assert "Trần Bình" in html and "Nguyễn Văn Tên Rất Dài Cần Xuống Dòng" not in html
    assert len(db.list_attendance(date.today().isoformat(), date.today().isoformat())) == 2


def test_edit_enable_save_reason_audit_and_refreshed_table(tmp_path):
    at, db = _app(tmp_path)
    next(x for x in at.button if x.label == "Lưu thay đổi").click().run()
    assert any("Bắt buộc nhập lý do" in x.value for x in at.error)
    assert not db.list_audit_logs()
    next(x for x in at.checkbox if x.label == "Đã check-out").check().run()
    out = next(x for x in at.time_input if x.label == "Giờ ra")
    assert not out.disabled
    out.set_value(time(17, 45))
    next(x for x in at.time_input if x.label == "Giờ vào").set_value(time(9, 0))
    next(x for x in at.selectbox if x.label == "Trạng thái").select("Đúng giờ")
    at.text_area[0].input("Đối chiếu dữ liệu kiểm thử")
    next(x for x in at.button if x.label == "Lưu thay đổi").click().run()
    assert not at.exception
    audits = db.list_audit_logs()
    assert len(audits) == 1
    values = json.loads(audits[0]["new_values"])
    assert values["check_in"].endswith("09:00:00")
    assert values["check_out"].endswith("17:45:00")
    assert values["status"] == "ON_TIME"
    assert "17:45:00" in " ".join(x.value for x in at.tabs[0].markdown)
    options = at.selectbox(key="attendance_edit_row").options
    other = next(option for option in options if option != at.selectbox(key="attendance_edit_row").value)
    at.selectbox(key="attendance_edit_row").select(other).run()
    assert not at.exception
    assert next(x for x in at.time_input if x.label == "Giờ vào").value == time(9, 22)
    assert not next(x for x in at.checkbox if x.label == "Đã check-out").value


def test_employee_cannot_open_admin_history_or_edit(tmp_path):
    path = tmp_path / "employee.db"
    source = f'''
import streamlit as st
from database.db import Database
from ui.history import render, _render_admin_editor
from auth.permissions import PermissionDenied
for action in (lambda: render(Database({str(path)!r}), "EMPLOYEE"),
               lambda: _render_admin_editor([], None, "EMPLOYEE")):
    try:
        action()
    except PermissionDenied:
        st.info("Access denied")
'''
    at = AppTest.from_string(source).run()
    assert not at.exception
    assert not at.tabs and not at.button
    assert len(at.info) == 2
