from __future__ import annotations

from html import escape

import streamlit as st

from auth.permissions import require_permission
from config import settings
from database.db import Database
from google_sheets.sync_service import SyncService
from ui.components import page_header, section_header


def render(db: Database, sync_service: SyncService, role: str) -> None:
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

    section_header("Đồng bộ Google Sheets", "Bản sao báo cáo; database chính luôn là nguồn dữ liệu gốc")
    source_name = "PostgreSQL" if db.is_postgres else "SQLite (chế độ chuyển tiếp)"
    st.caption(f"Nguồn dữ liệu chính: {source_name}. Google Sheets chỉ phục vụ báo cáo.")
    counts = db.sync_counts()
    synced = counts.get("SYNCED", 0)
    pending = counts.get("PENDING", 0)
    errors = counts.get("ERROR", 0)
    columns = st.columns(3)
    columns[0].metric("Đã đồng bộ", synced)
    columns[1].metric("Đang chờ", pending)
    columns[2].metric("Lỗi", errors)
    if not sync_service.client.configured:
        st.info(
            "Google Sheets chưa được cấu hình. Dữ liệu vẫn được lưu đầy đủ "
            "trong database chính; việc điểm danh không bị ảnh hưởng."
        )
    elif errors:
        st.warning(f"Có {errors} bản ghi chưa được đồng bộ Google Sheets.")
    sync_label = "Đồng bộ lại" if errors else "Đồng bộ Google Sheets"
    if st.button(sync_label, icon=":material/sync:", disabled=not sync_service.client.configured):
        require_permission(role, "sync.manage")
        with st.spinner("Đang đồng bộ báo cáo..."):
            result = sync_service.sync_pending(actor_role=role)
        if result.failed:
            st.warning(f"Đã đồng bộ {result.synced}; còn {result.failed} bản ghi lỗi. Dữ liệu điểm danh vẫn an toàn.")
        else:
            st.success(f"Đã đồng bộ {result.synced} bản ghi.")


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
