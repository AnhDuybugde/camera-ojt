"""Authentication views for the Streamlit application."""
from __future__ import annotations

from pathlib import Path
import time

import streamlit as st

from auth.permissions import ADMIN, EMPLOYEE
from auth.service import AuthService


ROLE_LABELS = {EMPLOYEE: "Nhân viên", ADMIN: "Quản trị viên"}


def render(auth_service: AuthService) -> None:
    st.markdown(
        """<style>
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display:none!important}
        .stAppViewContainer > .main {background:linear-gradient(145deg,#f6f8ff 0%,#eef3ff 48%,#f8fafc 100%)}
        </style>""",
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1, 1.05, 1], gap="large")
    del left, right
    with center:
        with st.container(key="login_card"):
            logo_path = Path(__file__).resolve().parents[1] / "static" / "logo-ai-mind.jpg"
            if logo_path.is_file():
                st.image(str(logo_path), width=76)
            else:
                st.markdown('<div class="login-logo-placeholder">AI</div>', unsafe_allow_html=True)
            st.markdown(
                """<div class="login-heading">
                <h1>Hệ thống chấm công</h1>
                <p>AI Mind JSC</p>
                <span>Đăng nhập để tiếp tục</span>
                </div>""",
                unsafe_allow_html=True,
            )
            selected_label = st.segmented_control(
                "Vai trò",
                [ROLE_LABELS[EMPLOYEE], ROLE_LABELS[ADMIN]],
                default=ROLE_LABELS[EMPLOYEE],
                width="stretch",
                key="login_role",
            )
            with st.form("login_form"):
                employee_id = ""
                if selected_label != ROLE_LABELS[ADMIN]:
                    employee_id = st.text_input(
                        "Tên đăng nhập", placeholder="Nhập mã nhân viên",
                        help="Nhân viên dùng mã NV làm tên đăng nhập.",
                    )
                password = st.text_input(
                    "Mật khẩu", type="password", placeholder="Nhập mật khẩu"
                )
                submitted = st.form_submit_button(
                    "Đăng nhập", type="primary", width="stretch", icon=":material/login:"
                )
            if submitted:
                role = ADMIN if selected_label == ROLE_LABELS[ADMIN] else EMPLOYEE
                try:
                    account = auth_service.login(role, password, employee_id.strip())
                except ValueError as exc:
                    st.error(str(exc))
                    return
                if account:
                    st.session_state.clear()
                    st.session_state.authenticated = True
                    st.session_state.role = role
                    st.session_state.employee_id = employee_id.strip() if role == EMPLOYEE else None
                    st.session_state.account_version = account.get("version")
                    st.session_state._last_activity_at = time.time()
                    st.session_state.active_page = "Tổng quan"
                    st.rerun()
                st.error("Tên đăng nhập hoặc mật khẩu không chính xác")


def render_account_controls(auth_service: AuthService, role: str) -> None:
    with st.expander("Bảo mật · Đổi mật khẩu", icon=":material/lock:"):
        with st.form("change_password_form", clear_on_submit=True):
            current = st.text_input("Mật khẩu hiện tại", type="password")
            new = st.text_input("Mật khẩu mới", type="password")
            confirmation = st.text_input("Xác nhận mật khẩu mới", type="password")
            change = st.form_submit_button("Cập nhật mật khẩu", width="stretch")
        if change:
            try:
                if role == EMPLOYEE:
                    auth_service.change_employee_password(st.session_state.employee_id, current, new, confirmation)
                    st.session_state.account_version = auth_service.account(st.session_state.employee_id)["version"]
                else:
                    auth_service.change_password(role, current, new, confirmation)
                st.success("Đã đổi mật khẩu thành công.")
            except ValueError as exc:
                st.error(str(exc))

    if st.button("Đăng xuất", icon=":material/logout:", width="stretch", key="logout"):
        for engine_group in ("live_engines", "spatial_engines"):
            for engine in st.session_state.get(engine_group, {}).values():
                try:
                    engine.stop()
                except Exception:
                    pass
        st.session_state.clear()
        st.rerun()


def force_password_change(auth_service: AuthService) -> None:
    st.warning("Bạn cần đổi mật khẩu ban đầu trước khi sử dụng hệ thống.")
    render_account_controls(auth_service, EMPLOYEE)
    if not auth_service.account(st.session_state.employee_id)["must_change"]:
        st.rerun()
    st.stop()
