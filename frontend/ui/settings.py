from __future__ import annotations

from html import escape

import streamlit as st

from auth.permissions import require_permission
from config import settings
from ui.components import page_header, section_header


def render(role: str) -> None:
    require_permission(role, "settings.view")
    page_header(
        "Cài đặt hệ thống",
        "Cấu hình vận hành của Hệ thống chấm công AI Mind JSC",
    )
    st.info("Sau khi chỉnh sửa tệp .env, hãy khởi động lại ứng dụng để áp dụng cấu hình mới.")
    section_header("Cấu hình đang áp dụng")

    groups = (
        ("Camera", "bi-camera-video", {
            "Camera điểm danh": (
                f"{len(settings.camera_sources)} luồng: "
                + ", ".join(name for name, _ in settings.camera_sources)
            ),
            "Nguồn đăng ký khuôn mặt": _safe_camera_label(),
            "Chu kỳ xử lý khung hình": str(settings.process_every_n_frames),
        }),
        ("Giờ làm việc", "bi-clock-history", {
            "Buổi sáng": f"{settings.work_start_time:%H:%M} - {settings.morning_end_time:%H:%M}",
            "Nghỉ trưa": f"{settings.morning_end_time:%H:%M} - {settings.afternoon_start_time:%H:%M}",
            "Buổi chiều": f"{settings.afternoon_start_time:%H:%M} - {settings.work_end_time:%H:%M}",
            "Mốc đi muộn": f"Sau {settings.late_threshold:%H:%M}",
        }),
        ("Điểm danh và đồng bộ", "bi-clipboard2-check", {
            "Chống ghi trùng": f"{settings.attendance_cooldown} giây",
            "Check-out tối thiểu": f"{settings.checkout_min_minutes} phút",
            "Ngưỡng nhận diện": str(settings.face_threshold),
            "Google Sheets": (
                "Đã cấu hình"
                if settings.google_sheet_id and settings.google_credentials_path
                else "Chưa cấu hình"
            ),
        }),
    )
    for column, group in zip(st.columns(3), groups):
        with column:
            _settings_card(*group)


def _safe_camera_label() -> str:
    for name, source in settings.camera_sources:
        if source == settings.registration_camera_source:
            return name
    return "Nguồn camera tùy chỉnh"


def _settings_card(title: str, icon_class: str, values: dict[str, str]) -> None:
    items = "".join(
        f'<div class="stitch-setting-item">'
        f'  <div class="stitch-setting-label">{escape(label)}</div>'
        f'  <div class="stitch-setting-value">{escape(value)}</div>'
        f'</div>'
        for label, value in values.items()
    )
    st.markdown(
        f'<div class="stitch-card">'
        f'  <div class="stitch-settings-title">'
        f'    <i class="bi {icon_class}" aria-hidden="true"></i>{escape(title)}'
        f'  </div>{items}'
        f'</div>',
        unsafe_allow_html=True,
    )
