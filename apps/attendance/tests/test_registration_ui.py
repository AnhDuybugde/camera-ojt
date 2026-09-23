from streamlit.testing.v1 import AppTest


def test_employee_schedule_only_own(tmp_path):
    source = '''
import streamlit as st
from database.db import Database
from ui.work_schedule import render
db = Database(PATH)
if not db.get_employee("NV001"):
    db.add_employee({"employee_id":"NV001", "full_name":"Own person"})
    db.add_employee({"employee_id":"NV002", "full_name":"Other person"})
st.session_state.role="EMPLOYEE"
st.session_state.employee_id="NV001"
render(db,"EMPLOYEE")
'''.replace("PATH", repr(str(tmp_path / "ui.db")))
    at = AppTest.from_string(source).run()
    assert not at.exception
    markup = " ".join(x.value for x in at.markdown)
    assert "Own person" in markup
    assert "Other person" not in markup
    assert "Tổng nhân sự công ty" not in markup
    assert "Tổng hợp On-site / Buổi" not in markup


def test_personal_attendance_empty_and_private(tmp_path):
    source = '''
import streamlit as st
from database.db import Database
from ui.my_attendance import render
db = Database(PATH)
if not db.get_employee("NV001"):
    db.add_employee({"employee_id":"NV001", "full_name":"Own person"})
    db.add_employee({"employee_id":"NV002", "full_name":"Other person"})
st.session_state.role="EMPLOYEE"
st.session_state.employee_id="NV001"
render(db,"EMPLOYEE")
'''.replace("PATH", repr(str(tmp_path / "attendance-ui.db")))
    at = AppTest.from_string(source).run()
    assert not at.exception
    assert any("Chưa ghi nhận CHECK-IN hôm nay" in x.value for x in at.markdown)
    assert not at.get("camera_input")
