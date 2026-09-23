from __future__ import annotations

import sqlite3

import streamlit as st

from auth.permissions import ADMIN, require_permission
from database.db import Database
from ui.components import (
    TableColumn, data_table, employee_cell, kpi_card, page_header, section_header,
    status_badge, translate_message,
)


def render(db: Database, role: str) -> None:
    require_permission(role, "employee.view")
    employees = db.list_employees()
    registered  = sum(bool(row["has_face"]) for row in employees)
    departments = len({row["department"] for row in employees if row["department"]})

    page_header(
        "Quản lý nhân viên",
        "Quản lý hồ sơ nhân sự và trạng thái dữ liệu khuôn mặt",
    )

    # KPI cards – Stitch colors
    for column, card in zip(st.columns(3), (
        ("Tổng nhân viên",       len(employees), "Hồ sơ đang quản lý",
         "groups",        "#004ac6", "#dce9ff"),
        ("Đã đăng ký khuôn mặt", registered,    "Sẵn sàng nhận diện",
         "face",          "#006c49", "rgba(108,248,187,0.30)"),
        ("Phòng ban",            departments,    "Đơn vị có nhân sự",
         "corporate_fare","#6d28d9", "#ede9fe"),
    )):
        with column:
            kpi_card(*card)

    st.write("")
    section_header("Danh sách nhân viên", "Dữ liệu trực tiếp từ SQLite")
    with st.container(key="ds_filter_employees"):
        query = st.text_input(
            "Tìm kiếm nhân viên",
            placeholder="Nhập mã, họ tên, phòng ban hoặc chức vụ…",
            icon=":material/search:",
        )
    normalized = query.strip().casefold()
    displayed = [
        row for row in employees
        if not normalized or normalized in " ".join(
            str(row.get(f, "")) for f in ("employee_id", "full_name", "department", "position")
        ).casefold()
    ]
    if displayed:
        data_table(
            [
                TableColumn("employee", "Nhân viên", 245),
                TableColumn("department", "Phòng ban", 120),
                TableColumn("position", "Chức vụ", 130),
                TableColumn("email", "Email", 190),
                TableColumn("face", "Khuôn mặt", 130),
                TableColumn("created", "Ngày tạo", 105),
                TableColumn("action", "Thao tác", 125),
            ],
            [{
                "employee": employee_cell(
                    row["full_name"], row["employee_id"], row["position"] or "Nhân viên"
                ),
                "department": row["department"] or "—",
                "position": row["position"] or "—",
                "email": row["email"] or "—",
                "face": status_badge(
                    "Đã đăng ký" if row["has_face"] else "Chưa đăng ký",
                    "success" if row["has_face"] else "warning",
                ),
                "created": row["created_date"],
                "action": status_badge(
                    "Sửa / Xóa bên dưới" if role == ADMIN else "Chỉ xem",
                    "primary" if role == ADMIN else "neutral",
                ),
            } for row in displayed],
            max_height=430,
        )
    elif employees:
        data_table([], [], empty_message="Không tìm thấy nhân viên phù hợp.")
    else:
        data_table([], [], empty_message="Chưa có nhân viên. Hãy thêm hồ sơ đầu tiên ở bên dưới.")

    tabs = st.tabs(
        ["Thêm nhân viên", "Chỉnh sửa hồ sơ", "Xóa nhân viên"]
        if role == ADMIN else ["Thêm nhân viên"]
    )
    add_tab = tabs[0]
    with add_tab:
        with st.form("add_employee", clear_on_submit=True):
            employee_id = st.text_input("Mã nhân viên *", placeholder="NV001")
            values = _fields("add")
            if st.form_submit_button(
                "Thêm nhân viên", type="primary", icon=":material/person_add:"
            ):
                try:
                    require_permission(role, "employee.add")
                    db.add_employee({"employee_id": employee_id, **values}, actor_role=role)
                    st.success("Đã thêm nhân viên.")
                    st.rerun()
                except (ValueError, sqlite3.IntegrityError) as exc:
                    st.error(f"Không thể thêm nhân viên: {translate_message(str(exc))}")

    if role != ADMIN:
        st.caption("Vai trò Nhân viên được thêm hồ sơ mới nhưng không có quyền sửa hoặc xóa hồ sơ đã tồn tại.")
        return

    edit_tab, delete_tab = tabs[1], tabs[2]
    ids = [row["employee_id"] for row in employees]
    with edit_tab:
        if ids:
            selected = st.selectbox("Chọn nhân viên", ids, key="edit_employee_id")
            current = next(row for row in employees if row["employee_id"] == selected)
            with st.form("edit_employee"):
                values = _fields(f"edit_{selected}", current)
                if st.form_submit_button(
                    "Lưu thay đổi", type="primary", icon=":material/save:"
                ):
                    try:
                        db.update_employee(selected, values, actor_role=role)
                        st.success("Đã cập nhật hồ sơ.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(translate_message(str(exc)))
        else:
            st.caption("Cần thêm nhân viên trước khi chỉnh sửa.")

    with delete_tab:
        if ids:
            selected = st.selectbox("Chọn nhân viên", ids, key="delete_employee_id")
            confirm = st.checkbox(
                "Tôi hiểu thao tác này cũng xóa lịch sử điểm danh của nhân viên."
            )
            if st.button(
                "Xóa nhân viên",
                disabled=not confirm,
                icon=":material/delete:",
            ):
                db.delete_employee(selected, actor_role=role)
                st.success("Đã xóa nhân viên.")
                st.rerun()


def _fields(prefix: str, current: dict | None = None) -> dict[str, str]:
    current = current or {}
    left, right = st.columns(2)
    return {
        "full_name":  left.text_input("Họ và tên *",    value=current.get("full_name", ""),   key=f"{prefix}_name"),
        "department": right.text_input("Phòng ban",     value=current.get("department", ""), key=f"{prefix}_department"),
        "position":   left.text_input("Chức vụ",        value=current.get("position", ""),   key=f"{prefix}_position"),
        "email":      right.text_input("Email",          value=current.get("email", ""),      key=f"{prefix}_email"),
        "phone":      left.text_input("Số điện thoại",  value=current.get("phone", ""),      key=f"{prefix}_phone"),
    }
