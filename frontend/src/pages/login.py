"""Authentication views for the Streamlit application."""
from __future__ import annotations

from config import settings

import streamlit as st

def _establish_session(account, email):
    st.session_state.clear()
    st.session_state.api_token = account.get("token", "")
    st.session_state.authenticated = True
    st.session_state.role = account["role"]
    st.session_state.employee_id = account.get("employee_id") or None
    st.session_state.email = account.get("email", email.strip().lower())
    st.session_state.active_page = "Tổng quan"
    st.rerun()


def render(auth_service) -> None:
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
            logo_path = settings.base_dir / "static" / "logo-ai-mind.jpg"
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
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="name@company.com")
                password = st.text_input(
                    "Mật khẩu", type="password", placeholder="Nhập mật khẩu"
                )
                submitted = st.form_submit_button(
                    "Đăng nhập", type="primary", width="stretch", icon=":material/login:"
                )
            if submitted:
                try:
                    account = auth_service.login(email.strip(), password)
                except ValueError as exc:
                    st.error(str(exc))
                    return
                if account:
                    _establish_session(account, email)
                st.error("Tên đăng nhập hoặc mật khẩu không chính xác")
            with st.expander("Kích hoạt tài khoản hoặc đăng nhập bằng mã email"):
                otp_email = st.text_input("Email tài khoản", key="otp_email")
                otp_code = st.text_input("Mã xác thực", key="otp_code")
                otp_purpose = st.selectbox(
                    "Mục đích", ["invite", "email"],
                    format_func=lambda value: "Kích hoạt lời mời" if value == "invite" else "Đăng nhập",
                    key="otp_purpose",
                )
                new_password = st.text_input("Tạo mật khẩu (bắt buộc khi kích hoạt)",
                                             type="password", key="invite_new_password")
                confirm_password = st.text_input("Xác nhận mật khẩu",
                                                 type="password", key="invite_confirm_password")
                send_code, verify_code = st.columns(2)
                if send_code.button("Gửi mã email", key="send_otp"):
                    try:
                        auth_service.request_otp(otp_email.strip())
                        st.success("Nếu tài khoản hợp lệ, mã đã được gửi.")
                    except (ValueError, RuntimeError) as exc:
                        st.error(str(exc))
                if verify_code.button("Xác thực", key="verify_otp"):
                    try:
                        account = auth_service.verify_otp(
                            otp_email.strip(), otp_code.strip(), otp_purpose,
                            new_password, confirm_password,
                        )
                        _establish_session(account, otp_email)
                    except (ValueError, RuntimeError) as exc:
                        st.error(str(exc))
            with st.expander("Quên mật khẩu?"):
                recover_email = st.text_input("Email khôi phục", key="recover_email")
                if st.button("Gửi hướng dẫn đặt lại mật khẩu"):
                    try:
                        auth_service.recover(recover_email.strip())
                        st.success("Nếu email tồn tại, mã khôi phục đã được gửi.")
                    except (ValueError, RuntimeError) as exc:
                        st.error(str(exc))
                recovery_code = st.text_input("Mã khôi phục", key="recovery_code")
                reset_password = st.text_input("Mật khẩu mới", type="password", key="recovery_password")
                reset_confirmation = st.text_input("Xác nhận mật khẩu mới", type="password",
                                                   key="recovery_confirmation")
                if st.button("Đặt lại mật khẩu", key="finish_recovery"):
                    try:
                        account = auth_service.verify_otp(
                            recover_email.strip(), recovery_code.strip(), "recovery",
                            reset_password, reset_confirmation,
                        )
                        _establish_session(account, recover_email)
                    except (ValueError, RuntimeError) as exc:
                        st.error(str(exc))


def render_account_controls(auth_service, role: str) -> None:
    with st.expander("Bảo mật · Đổi mật khẩu", icon=":material/lock:"):
        with st.form("change_password_form", clear_on_submit=True):
            current = st.text_input("Mật khẩu hiện tại", type="password")
            new = st.text_input("Mật khẩu mới", type="password")
            confirmation = st.text_input("Xác nhận mật khẩu mới", type="password")
            change = st.form_submit_button("Cập nhật mật khẩu", width="stretch")
        if change:
            try:
                auth_service.change_password(current, new, confirmation)
                st.success("Đã đổi mật khẩu thành công.")
            except (ValueError, RuntimeError) as exc:
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
