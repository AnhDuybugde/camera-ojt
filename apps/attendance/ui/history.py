from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pandas as pd
import streamlit as st

from attendance.admin_service import AttendanceAdminService
from auth.permissions import ADMIN, PermissionDenied, require_permission
from database.db import Database
from ui.components import (
    TableColumn, data_table, employee_cell, page_header, section_header, status_badge,
)


def render(
    db: Database, role: str, admin_service: AttendanceAdminService | None = None
) -> None:
    require_permission(role, "attendance.view")
    page_header(
        "Lịch sử điểm danh",
        "Tra cứu, lọc và xuất dữ liệu chấm công theo thời gian",
    )
    employees = db.list_employees()
    departments = sorted({row["department"] for row in employees if row["department"]})

    with st.container(key="ds_filter_history"):
        section_header("Bộ lọc dữ liệu")
        col1, col2, col3, col4 = st.columns(4)
        date_from = col1.date_input("Từ ngày", date.today() - timedelta(days=30))
        date_to = col2.date_input("Đến ngày", date.today())
        employee = col3.selectbox(
            "Nhân viên", ["Tất cả"] + [row["employee_id"] for row in employees]
        )
        department = col4.selectbox("Phòng ban", ["Tất cả"] + departments)

    rows = db.list_attendance(
        date_from.isoformat(), date_to.isoformat(),
        None if employee == "Tất cả" else employee,
        None if department == "Tất cả" else department,
    )
    section_header("Kết quả tra cứu", f"{len(rows)} bản ghi phù hợp")
    if rows:
        frame = pd.DataFrame([{
            "Mã NV": row["employee_id"],
            "Nhân viên": row["employee_name"],
            "Phòng ban": row["department"] or "—",
            "Ngày": row["date"],
            "Giờ vào": _clock(row["check_in"]),
            "Giờ ra": _clock(row["check_out"]),
            "Trạng thái": _status_tags(row),
        } for row in rows])
        data_table(
            [
                TableColumn("employee", "Nhân viên", 260),
                TableColumn("employee_id", "Mã NV", 110),
                TableColumn("department", "Phòng ban", 140),
                TableColumn("date", "Ngày", 125),
                TableColumn("check_in", "Giờ vào", 110),
                TableColumn("check_out", "Giờ ra", 110),
                TableColumn("status", "Trạng thái", 180),
            ],
            [{
                "employee": employee_cell(
                    row["employee_name"], row["employee_id"],
                    row["department"] or "Nhân viên", show_id=False,
                ),
                "employee_id": row["employee_id"],
                "department": row["department"] or "—",
                "date": row["date"],
                "check_in": _clock(row["check_in"]),
                "check_out": _clock(row["check_out"]),
                "status": _status_badges(row),

            } for row in rows],
            max_height=470,
        )
        export = frame.copy()
        export["Trạng thái"] = export["Trạng thái"].map(" · ".join)
        st.download_button(
            "Tải xuống CSV", export.to_csv(index=False).encode("utf-8-sig"),
            "lich_su_diem_danh.csv", "text/csv", icon=":material/download:",
        )
    else:
        data_table([], [], empty_message="Không có bản ghi phù hợp với bộ lọc.")

    if role == ADMIN and rows and admin_service is not None:
        _render_admin_editor(rows, admin_service, role)


def _clock(value: str | None) -> str:
    return value.split("T", 1)[-1][:8] if value else "—"


def _status_tags(row: dict) -> list[str]:
    labels = {"ON_TIME": "Đúng giờ", "LATE": "Đi muộn", "ABSENT": "Vắng"}
    tags = [labels.get(row["status"], row["status"])]
    if row.get("check_out"):
        tags.append("Đã check-out")
    return tags


def _status_badges(row: dict) -> list:
    tone = {"ON_TIME": "success", "LATE": "warning", "ABSENT": "danger"}.get(
        row["status"], "neutral"
    )
    badges = [status_badge(_status_tags(row)[0], tone)]
    if row.get("check_out"):
        badges.append(status_badge("Đã check-out", "info"))
    return badges


def _render_admin_editor(
    rows: list[dict], service: AttendanceAdminService, role: str
) -> None:
    st.write("")
    section_header(
        "Chỉnh sửa chấm công",
        "Chỉ Quản trị viên · mọi thay đổi đều được lưu vào nhật ký",
    )
    labels = {
        f'{row["date"]} · {row["employee_id"]} · {row["employee_name"]} · #{row["id"]}': row
        for row in rows
    }
    current = labels[st.selectbox("Chọn bản ghi", list(labels), key="attendance_edit_row")]
    current_date = date.fromisoformat(current["date"])
    current_in = _to_time(current.get("check_in"), time(9, 0))
    current_out = _to_time(current.get("check_out"), time(17, 30))
    with st.form("attendance_admin_edit"):
        left, middle, right = st.columns(3)
        work_date = left.date_input("Ngày", current_date)
        check_in = middle.time_input("Giờ vào", current_in)
        has_checkout = right.checkbox("Đã check-out", value=bool(current.get("check_out")))
        checkout_time = right.time_input("Giờ ra", current_out, disabled=not has_checkout)
        status_label = st.selectbox(
            "Trạng thái", ["Tự động", "Đúng giờ", "Đi muộn", "Vắng"],
            help="Tự động sẽ tính lại theo mốc đi muộn trong cấu hình hệ thống.",
        )
        reason = st.text_area(
            "Lý do chỉnh sửa *", placeholder="Ví dụ: Nhân viên quên check-out"
        )
        submitted = st.form_submit_button(
            "Lưu thay đổi", type="primary", icon=":material/save:"
        )
    if submitted:
        status_map = {"Đúng giờ": "ON_TIME", "Đi muộn": "LATE", "Vắng": "ABSENT"}
        try:
            service.update_attendance(
                int(current["id"]), actor_role=role, work_date=work_date.isoformat(),
                check_in=datetime.combine(work_date, check_in).isoformat(timespec="seconds"),
                check_out=(
                    datetime.combine(work_date, checkout_time).isoformat(timespec="seconds")
                    if has_checkout else None
                ),
                reason=reason, status_override=status_map.get(status_label),
            )
            st.success("Đã cập nhật chấm công và ghi nhật ký chỉnh sửa.")
            st.rerun()
        except (ValueError, PermissionDenied) as exc:
            st.error(str(exc))


def _to_time(value: str | None, fallback: time) -> time:
    if not value:
        return fallback
    return datetime.fromisoformat(value).time().replace(microsecond=0)
