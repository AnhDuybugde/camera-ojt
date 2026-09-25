"""Administrator audit-log page."""
from __future__ import annotations

import json

import streamlit as st

from auth.permissions import require_permission
from database.db import Database
from frontend.src.components.components import TableColumn, data_table, page_header, section_header, status_badge


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
            TableColumn("timestamp", "Thời gian", 155),
            TableColumn("role", "Người thực hiện", 125),
            TableColumn("action", "Hành động", 125),
            TableColumn("employee", "Đối tượng", 95),
            TableColumn("old", "Giá trị cũ", 270),
            TableColumn("new", "Giá trị mới", 270),
            TableColumn("reason", "Lý do", 220),
        ],
        [{
            "timestamp": row["timestamp"].replace("T", " ")[:19],
            "role": status_badge(
                "Quản trị viên" if row["role"] == "ADMIN" else row["role"], "info"
            ),
            "action": status_badge(
                "Sửa chấm công" if row["action"] == "UPDATE_ATTENDANCE" else row["action"],
                "primary",
            ),
            "employee": row["employee_id"],
            "old": _summary(row["old_values"]),
            "new": _summary(row["new_values"]),
            "reason": row["reason"],
        } for row in rows],
        max_height=560,
    )


def _summary(raw: str) -> str:
    try:
        values = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return str(raw)
    return " · ".join(f"{key}: {value if value is not None else '—'}" for key, value in values.items())
