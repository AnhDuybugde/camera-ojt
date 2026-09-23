from __future__ import annotations

import streamlit as st

from auth.permissions import require_permission
from database.db import Database
from google_sheets.sync_service import SyncService
from ui.components import (
    TableColumn, data_table, employee_cell, kpi_card, page_header, section_header,
    status_badge, translate_message,
)


def render(db: Database, service: SyncService, role: str) -> None:
    require_permission(role, "sync.manage")
    pending = db.pending_attendance()
    page_header(
        "Đồng bộ Google Sheets",
        "Theo dõi hàng đợi và chủ động đồng bộ dữ liệu báo cáo",
    )

    for column, card in zip(st.columns(2), (
        ("Đang chờ / lỗi", len(pending), "Bản ghi cần xử lý",
         "sync", "#784b00", "#ffddb8"),
        ("Nguồn dữ liệu", "SQLite", "Luôn là dữ liệu chính",
         "database", "#004ac6", "#dce9ff"),
    )):
        with column:
            kpi_card(*card)

    st.write("")
    section_header("Hàng đợi đồng bộ", "Google Sheets chỉ phục vụ báo cáo")
    if pending:
        data_table(
            [
                TableColumn("employee", "Nhân viên", 230),
                TableColumn("date", "Ngày", 105),
                TableColumn("check_in", "Giờ vào", 150),
                TableColumn("check_out", "Giờ ra", 150),
                TableColumn("attendance", "Chấm công", 110),
                TableColumn("sync", "Đồng bộ", 105),
            ],
            [{
                "employee": employee_cell(
                    row["employee_name"], row["employee_id"], row["department"] or "Nhân viên"
                ),
                "date": row["date"],
                "check_in": row["check_in"],
                "check_out": row["check_out"] or "—",
                "attendance": status_badge(
                    "Đúng giờ" if row["status"] == "ON_TIME" else "Đi muộn",
                    "success" if row["status"] == "ON_TIME" else "warning",
                ),
                "sync": status_badge(
                    row["sync_status"],
                    "warning" if row["sync_status"] == "PENDING" else "danger",
                ),
            } for row in pending],
            max_height=430,
        )
    else:
        st.success("Không có bản ghi đang chờ đồng bộ.")

    if st.button("Đồng bộ ngay", type="primary", icon=":material/sync:"):
        with st.spinner("Đang đồng bộ dữ liệu…"):
            result = service.sync_pending(actor_role=role)
        message = f"Đã đồng bộ {result.synced} bản ghi; lỗi {result.failed} bản ghi."
        if result.synced == 0 and result.failed == 0:
            message = translate_message(result.message)
        elif "not configured" in result.message:
            message = translate_message(
                "Google Sheets is not configured; records remain pending."
            )
        (st.success if result.failed == 0 else st.warning)(message)
