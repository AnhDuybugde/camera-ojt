from __future__ import annotations

from datetime import date, datetime, time, timedelta
from html import escape
from pathlib import Path

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
    css = (Path(__file__).resolve().parents[1] / "assets" / "history.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    with st.container(key="attendance_history_page"):
        if notice := st.session_state.pop("history_save_notice", None):
            st.success(notice)
        if role == ADMIN:
            history_tab, edit_tab = st.tabs(["Lịch sử chấm công", "Chỉnh sửa chấm công"])
            with history_tab:
                rows = _render_history(db)
            with edit_tab:
                if not rows:
                    st.info("Không có bản ghi phù hợp. Hãy điều chỉnh bộ lọc trong tab Lịch sử chấm công.")
                elif admin_service is None:
                    st.info("Chức năng chỉnh sửa chưa sẵn sàng.")
                else:
                    _render_admin_editor(rows, admin_service, role)
        else:
            _render_history(db)


def _render_history(db: Database) -> list[dict]:
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
                TableColumn("employee", "Nhân viên", 280),
                TableColumn("employee_id", "Mã NV", 90),
                TableColumn("department", "Phòng ban", 110),
                TableColumn("date", "Ngày", 115),
                TableColumn("check_in", "Giờ vào", 100),
                TableColumn("check_out", "Giờ ra", 100),
                TableColumn("status", "Trạng thái", 220),
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

    return rows


def _clock(value: str | None) -> str:
    return value.split("T", 1)[-1][:8] if value else "—"


def _status_tags(row: dict) -> list[str]:
    labels = {"ON_TIME": "Đúng giờ", "LATE": "Đi muộn", "ABSENT": "Vắng mặt", "WAITING": "Chờ chấm công"}
    tags = [labels.get(row["status"], row["status"])]
    if row.get("presence_status") == "TEMP_CHECKOUT":
        tags.append("Check-out tạm thời")
    elif row.get("presence_status") == "ABSENT" and row.get("check_in"):
        tags.append("Đang vắng khỏi văn phòng")
    if row.get("check_out"):
        tags.append("Đã check-out")
    return tags


def _status_badges(row: dict) -> list:
    tone = {"ON_TIME": "success", "LATE": "warning", "ABSENT": "danger", "WAITING": "neutral"}.get(
        row["status"], "neutral"
    )
    badges = [status_badge(_status_tags(row)[0], tone)]
    if row.get("presence_status") == "TEMP_CHECKOUT":
        badges.append(status_badge("Check-out tạm thời", "warning"))
    elif row.get("presence_status") == "ABSENT" and row.get("check_in"):
        badges.append(status_badge("Đang vắng khỏi văn phòng", "danger"))
    if row.get("check_out"):
        badges.append(status_badge("Đã check-out", "info"))
    return badges


def _render_admin_editor(
    rows: list[dict], service: AttendanceAdminService, role: str
) -> None:
    require_permission(role, "attendance.update")
    section_header(
        "Chỉnh sửa chấm công",
        "Chỉ Quản trị viên · mọi thay đổi đều được lưu vào nhật ký",
    )
    labels = {
        f'{row["date"]} · {row["employee_id"]} · {row["employee_name"]} · #{row["id"]}': row
        for row in rows
    }
    with st.container(key="history_record_card"):
        current = labels[st.selectbox("Chọn bản ghi", list(labels), key="attendance_edit_row")]
        name = current["employee_name"]
        initials = "".join(word[0].upper() for word in name.split()[-2:]) or "NV"
        st.markdown(
            '<div class="history-snapshot"><div class="history-identity">'
            f'<span class="history-avatar">{escape(initials)}</span><div><b>{escape(name)}</b>'
            f'<small>{escape(current["employee_id"])} · {escape(current["department"] or "—")}</small></div></div>'
            f'<div><small>Bản ghi đang chọn</small><b>#{current["id"]} · {escape(current["date"])}</b></div>'
            f'<div><small>Dữ liệu hiện tại</small><b>Vào: {_clock(current["check_in"])} · Ra: {_clock(current["check_out"])}</b></div>'
            f'<div><small>Trạng thái hiện tại</small><b>{escape(" · ".join(_status_tags(current)))}</b></div></div>',
            unsafe_allow_html=True,
        )
    current_date = date.fromisoformat(current["date"])
    current_in = _to_time(current.get("check_in"), time(9, 0))
    current_out = _to_time(current.get("check_out"), time(17, 30))
    # Per-record keys prevent drafts leaking between selected records. Widgets
    # rerun locally so checkboxes immediately enable/disable their time inputs.
    token = f'history_edit_{current["id"]}_{current.get("updated_at", "")}'
    with st.container(key="history_edit_card"):
        st.markdown('<div class="history-edit-heading">Chi tiết điều chỉnh chấm công</div>', unsafe_allow_html=True)
        left, middle, right = st.columns(3)
        work_date = left.date_input("Ngày", current_date, key=f"{token}_date", format="DD/MM/YYYY")
        with middle.container(key="history_checkin_card"):
            has_check_in = st.checkbox("Đã check-in", value=bool(current.get("check_in")), key=f"{token}_has_in")
            check_in = st.time_input("Giờ vào", current_in, disabled=not has_check_in, key=f"{token}_in")
        with right.container(key="history_checkout_card"):
            has_checkout = st.checkbox("Đã check-out", value=bool(current.get("check_out")), key=f"{token}_has_out")
            checkout_time = st.time_input("Giờ ra", current_out, disabled=not has_checkout, key=f"{token}_out")
        status_label = st.selectbox(
            "Trạng thái", ["Tự động", "Đúng giờ", "Đi muộn", "Vắng"],
            help="Tự động sẽ tính lại theo mốc đi muộn trong cấu hình hệ thống.",
            key=f"{token}_status",
        )
        reason = st.text_area(
            "Lý do chỉnh sửa *", placeholder="Ví dụ: Nhân viên quên check-out",
            key=f"{token}_reason", height=110,
        )
        with st.container(key="history_save_bar"):
            note, action = st.columns([3, 1], vertical_alignment="center")
            note.caption("Lý do là bắt buộc. Mọi thay đổi được ghi vào nhật ký chỉnh sửa.")
            submitted = action.button("Lưu thay đổi", type="primary", icon=":material/save:", key=f"{token}_save", width="stretch")
    if submitted:
        status_map = {"Đúng giờ": "ON_TIME", "Đi muộn": "LATE", "Vắng": "ABSENT"}
        try:
            service.update_attendance(
                int(current["id"]), actor_role=role, work_date=work_date.isoformat(),
                check_in=(datetime.combine(work_date, check_in).isoformat(timespec="seconds")
                          if has_check_in else None),
                check_out=(
                    datetime.combine(work_date, checkout_time).isoformat(timespec="seconds")
                    if has_checkout else None
                ),
                reason=reason, status_override=status_map.get(status_label),
            )
            st.session_state.history_save_notice = "Đã cập nhật chấm công và ghi nhật ký chỉnh sửa."
            st.rerun()
        except (ValueError, PermissionDenied) as exc:
            st.error(str(exc))


def _to_time(value: str | None, fallback: time) -> time:
    if not value:
        return fallback
    return datetime.fromisoformat(value).time().replace(microsecond=0)
