"""Streamlit entry point with role-based authentication and navigation."""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import streamlit as st

from auth.permissions import ADMIN, EMPLOYEE, require_permission
from config import settings
import ui.components as ui_components
from ui import (
    audit_log, dashboard, employees, history, live_attendance, login,
    register_face, settings as settings_page, spatial_analytics, statistics,
    work_schedule, my_attendance, zone_labeler,
)
from ui.common import (
    get_attendance_admin_service, get_attendance_service, get_auth_service, get_db,
    get_daily_attendance_worker, get_detector, get_recognizer, get_sync_service, get_sync_worker,
)
from ui.theme import apply_theme
from utils.logger import setup_logging

setup_logging()
settings.ensure_directories()
st.set_page_config(
    page_title="Hệ thống chấm công AI Mind JSC",
    page_icon=":material/face:", layout="wide", initial_sidebar_state="expanded",
)
apply_theme()

# Refresh presentation-only modules once per browser session. This is required
# for long-running Streamlit processes, which may otherwise retain imported page
# modules even after their table implementation changes on disk.
_TABLE_UI_VERSION = "history-stitch-v11"
if st.session_state.get("_table_ui_version") != _TABLE_UI_VERSION:
    importlib.reload(ui_components)
    for page_module in (
        employees, history, dashboard, statistics, audit_log,
        live_attendance, spatial_analytics, work_schedule, register_face, login, my_attendance,
        zone_labeler,
    ):
        importlib.reload(page_module)
    st.session_state._table_ui_version = _TABLE_UI_VERSION

db = get_db(4)
get_daily_attendance_worker()
auth_service = get_auth_service(2)
role = st.session_state.get("role")
if role == EMPLOYEE and st.session_state.get("authenticated"):
    account = auth_service.account(st.session_state.get("employee_id", ""))
    if not account or account["version"] != st.session_state.get("account_version"):
        st.session_state.clear()
        role = None
if not st.session_state.get("authenticated") or role not in {EMPLOYEE, ADMIN}:
    login.render(auth_service)
    st.stop()

if role == EMPLOYEE and account["must_change"]:
    login.force_password_change(auth_service)

ALL_PAGES = (
    ("Tổng quan", "Dashboard", "grid_view", "dashboard"),
    ("Đăng ký", "Registration", "badge", "registration"),
    ("Lịch làm việc", "Work Schedule", "calendar_month", "work_schedule"),
    ("Điểm danh trực tiếp", "Live Attendance", "videocam", "live"),
    ("Gắn nhãn vùng", "Zone Labeler", "gesture", "zone_labeler"),
    ("Lịch sử chấm công", "Attendance History", "event_available", "history"),
    ("Nhật ký chỉnh sửa", "Audit Log", "history_edu", "audit"),
    ("Báo cáo & Phân tích", "Reports", "insights", "reports"),
    ("Cài đặt", "Settings", "settings", "settings"),
)
EMPLOYEE_PAGE_KEYS = {
    "Registration", "Work Schedule"
}
menu = [item for item in ALL_PAGES if role == ADMIN or item[1] in EMPLOYEE_PAGE_KEYS]
if role == EMPLOYEE:
    menu.insert(0, ("Chấm công của tôi", "My Attendance", "event_available", "my_attendance"))
    menu.sort(key=lambda item: {"My Attendance": 0, "Work Schedule": 1, "Registration": 2}[item[1]])
allowed_labels = {item[0] for item in menu}
if st.session_state.get("active_page") not in allowed_labels:
    st.session_state.active_page = "Tổng quan" if role == ADMIN else "Chấm công của tôi"


def _select_page(label: str) -> None:
    st.session_state.active_page = label


def _render_current(module: object, expected_args: int, *args: object) -> None:
    """Reload stale page modules left alive by Streamlit hot-reload.

    The authentication change added ``role`` to several render functions. An
    already-running Streamlit process can retain the previous module object
    while re-executing the new app.py, which otherwise raises a TypeError on
    every affected tab.
    """
    render = getattr(module, "render")
    if len(inspect.signature(render).parameters) != expected_args:
        render = getattr(importlib.reload(module), "render")
    render(*args)


with st.sidebar:
    logo_path = Path(__file__).resolve().parent / "static" / "logo-ai-mind.jpg"
    logo_html = (
        '<img src="app/static/logo-ai-mind.jpg" alt="Logo AI Mind JSC" />'
        if logo_path.is_file()
        else '<span class="stitch-brand-logo-placeholder" style="display:flex">AI</span>'
    )
    logo_class = "" if logo_path.is_file() else " is-placeholder"
    role_label = "Quản trị viên" if role == ADMIN else "Nhân viên"
    st.markdown(
        f"""<div class="stitch-brand">
          <div class="stitch-brand-logo{logo_class}">{logo_html}</div>
          <div class="stitch-brand-text">
            <div class="stitch-brand-name">Hệ thống chấm công</div>
            <div class="stitch-brand-sub">AI Mind JSC · {role_label}</div>
          </div>
        </div><div class="stitch-nav-label">Menu</div>""",
        unsafe_allow_html=True,
    )
    for label, _, icon, key in menu:
        st.button(
            label, key=f"nav_{key}", icon=f":material/{icon}:", width="stretch",
            type="primary" if st.session_state.active_page == label else "tertiary",
            on_click=_select_page, args=(label,),
        )
    st.markdown('<div class="stitch-nav-label">Tài khoản</div>', unsafe_allow_html=True)
    login.render_account_controls(auth_service, role)
    st.markdown(
        """<div class="stitch-sidebar-footer"><div class="stitch-version-chip">
        <span>Phiên bản v2.5.0 · Enterprise</span><span class="stitch-version-dot"></span>
        </div></div>""", unsafe_allow_html=True,
    )

selection = st.session_state.active_page
page = next(item[1] for item in menu if item[0] == selection)
attendance_service = get_attendance_service(4)
for live_engine in st.session_state.get("live_engines", {}).values():
    live_engine.attendance = attendance_service
get_sync_worker()

if page == "Dashboard":
    dashboard.render(db)
elif page == "Registration":
    st.markdown('<div class="ws-page-head"><div><div class="ws-title-line"><h1>Đăng ký</h1></div><p>Quản lý hồ sơ và đăng ký dữ liệu khuôn mặt</p></div></div>', unsafe_allow_html=True)
    registration_section = "Đăng ký khuôn mặt"
    if role == ADMIN:
        registration_section = st.segmented_control(
            "Chọn chức năng đăng ký", ["Nhân viên", "Đăng ký khuôn mặt"],
            default="Nhân viên", key="registration_section", width="stretch",
        ) or "Nhân viên"
    if role == ADMIN and registration_section == "Nhân viên":
        with st.container():
            employees.render(db, role)
            auth_service.ensure_accounts()
            rows = db.list_employees()
            if rows:
                with st.expander("Tài khoản nhân viên · Đặt lại mật khẩu"):
                    selected = st.selectbox("Mã nhân viên", [r["employee_id"] for r in rows], key="reset_account")
                    confirm = st.checkbox("Xác nhận đặt lại mật khẩu về 123 và yêu cầu đổi mật khẩu")
                    if st.button("Đặt lại mật khẩu", disabled=not confirm):
                        auth_service.reset_employee_password(selected, actor_role=role)
                        st.success("Đã đặt lại mật khẩu. Phiên đăng nhập cũ sẽ hết hiệu lực.")
    if registration_section == "Đăng ký khuôn mặt":
        register_face.render(db, get_detector(), get_recognizer(), role)
elif page == "Work Schedule":
    _render_current(work_schedule, 2, db, role)
elif page == "My Attendance":
    my_attendance.render(db, role)
elif page == "Live Attendance":
    _render_current(
        live_attendance, 4, get_detector(), get_recognizer(), attendance_service, role
    )
elif page == "Zone Labeler":
    _render_current(zone_labeler, 1, role)
elif page == "Attendance History":
    _render_current(history, 3, db, role, get_attendance_admin_service())
elif page == "Audit Log":
    _render_current(audit_log, 2, db, role)
elif page == "Reports":
    require_permission(role, "reports.view")
    st.markdown('<div class="ws-page-head"><div><div class="ws-title-line"><h1>Báo cáo &amp; Phân tích</h1></div><p>Báo cáo chấm công và phân tích không gian, ra vào</p></div></div>', unsafe_allow_html=True)
    report_section = st.segmented_control(
        "Chọn chức năng báo cáo", ["Chấm công", "Không gian & Ra vào"],
        default="Chấm công", key="report_section", width="stretch",
    ) or "Chấm công"
    if report_section == "Chấm công":
        statistics.render(db)
    else:
        st.caption("CHECK-IN/OUT là dữ liệu chấm công; mật độ và luồng di chuyển thực tế lấy từ camera phân tích.")
        _render_current(spatial_analytics, 4, db, role, get_detector(), get_recognizer())
else:
    _render_current(settings_page, 3, db, get_sync_service(), role)
