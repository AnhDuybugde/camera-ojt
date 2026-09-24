"""Administrator audit-log page."""
from __future__ import annotations

import json
from datetime import date, datetime

import streamlit as st

from auth.permissions import require_permission
from database.db import Database
from ui.components import TableColumn, data_table, page_header, section_header, status_badge


ROLE_LABELS = {
    "ADMIN": "Quản trị viên",
    "EMPLOYEE": "Nhân viên",
    "SYSTEM": "Hệ thống",
}

ACTION_LABELS = {
    "SCHEDULE_UPDATE": "Cập nhật lịch",
    "PASSWORD_RESET": "Đặt lại mật khẩu",
    "UPDATE_ATTENDANCE": "Sửa chấm công",
    "FACE_REGISTER": "Đăng ký khuôn mặt",
    "MARK_ABSENT": "Ghi nhận vắng",
    "TEMP_CHECKOUT": "Ra ngoài tạm thời",
    "RETURN_TO_WORK": "Quay lại làm việc",
    "PRESENCE_TIMEOUT": "Hết thời gian ra ngoài",
    "LATE_CHECK_IN": "Vào làm sau khi vắng",
}

FIELD_LABELS = {
    "date": "Ngày", "work_date": "Ngày làm việc",
    "check_in": "Giờ vào", "check_out": "Giờ ra",
    "status": "Trạng thái", "presence_status": "Hiện diện",
    "temporary_checkout_at": "Giờ ra ngoài", "returned_at": "Giờ quay lại",
    "work_session": "Buổi", "work_status": "Hình thức",
}

VALUE_LABELS = {
    "ADMIN": "Quản trị viên", "EMPLOYEE": "Nhân viên", "SYSTEM": "Hệ thống",
    "ON_TIME": "Đúng giờ", "LATE": "Đi muộn", "ABSENT": "Vắng mặt",
    "WAITING": "Đang chờ", "PRESENT": "Có mặt", "TEMP_CHECKOUT": "Ra ngoài tạm thời",
    "MORNING": "Buổi sáng", "AFTERNOON": "Buổi chiều",
    "ON": "Tại văn phòng", "WFH": "Làm từ xa", "OFF": "Nghỉ",
}


def render(db: Database, role: str) -> None:
    require_permission(role, "audit.view")
    page_header(
        "Nhật ký chỉnh sửa",
        "Theo dõi minh bạch các thay đổi thủ công đối với dữ liệu chấm công",
    )
    limit = st.selectbox("Số bản ghi gần nhất", [100, 250, 500], index=0)
    rows = db.list_audit_logs(limit)
    section_header("Lịch sử thao tác", f"{len(rows)} bản ghi")
    if not rows:
        data_table([], [], empty_message="Chưa có thao tác chỉnh sửa nào được ghi nhận.")
        return
    data_table(
        [
            TableColumn("timestamp", "Thời gian", 140),
            TableColumn("role", "Người thực hiện", 110),
            TableColumn("action", "Hành động", 135),
            TableColumn("employee", "Đối tượng", 85),
            TableColumn("old", "Giá trị cũ", 220),
            TableColumn("new", "Giá trị mới", 220),
            TableColumn("reason", "Lý do", 170),
        ],
        [{
            "timestamp": _timestamp(row["timestamp"]),
            "role": status_badge(
                ROLE_LABELS.get(row["role"], "Hệ thống"),
                "info" if row["role"] == "ADMIN" else "neutral",
            ),
            "action": status_badge(
                ACTION_LABELS.get(row["action"], "Thao tác khác"),
                "primary",
            ),
            "employee": row["employee_id"],
            "old": _summary(row["old_values"]),
            "new": _summary(row["new_values"]),
            "reason": row["reason"] or "—",
        } for row in rows],
        compact=True, max_height=560,
    )


def _summary(raw: str) -> str:
    try:
        values = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return str(raw)
    if not isinstance(values, dict) or not values:
        return "—"
    return " · ".join(
        f"{FIELD_LABELS.get(key, 'Thông tin')}: {_format_value(key, value)}"
        for key, value in values.items()
    )


def _format_value(key: str, value: object) -> str:
    if value is None or value == "":
        return "—"
    text = str(value)
    if text in VALUE_LABELS:
        return VALUE_LABELS[text]
    try:
        if key in {"date", "work_date"}:
            return date.fromisoformat(text[:10]).strftime("%d/%m/%Y")
        if key in {"check_in", "check_out", "temporary_checkout_at", "returned_at"}:
            return datetime.fromisoformat(text).strftime("%d/%m %H:%M")
    except ValueError:
        pass
    return text


def _timestamp(raw: str) -> str:
    try:
        return datetime.fromisoformat(raw).strftime("%d/%m/%Y %H:%M")
    except (TypeError, ValueError):
        return str(raw).replace("T", " ")[:16]
