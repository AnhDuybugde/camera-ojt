"""Stitch-faithful weekly work-schedule editor backed by SQLite data."""
from __future__ import annotations

from datetime import date, timedelta
from html import escape

import pandas as pd
import streamlit as st

from auth.permissions import EMPLOYEE, require_permission
from config import settings
from database.db import Database


UNSCHEDULED = "Chưa đăng ký"
DAY_NAMES = ("Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6")
WORK_SESSIONS = (("MORNING", "Sáng"), ("AFTERNOON", "Chiều"))
STATUS_OPTIONS = (UNSCHEDULED, "ON", "WFH", "OFF")
STATUS_META = {
    "ON": ("ON", ":material/apartment:"),
    "WFH": ("WFH", ":material/house:"),
    "OFF": ("OFF", ":material/coffee:"),
}
AVATAR_COLORS = ("#4f46e5", "#2563eb", "#059669", "#db2777", "#475569")


def _monday(value: date) -> date:
    return value - timedelta(days=value.weekday())


def _set_week(value: date) -> None:
    st.session_state.schedule_week_start = _monday(value)


def _initials(name: str) -> str:
    words = [part for part in name.split() if part]
    return "".join(part[0].upper() for part in words[-2:]) or "NV"


def _draft_key(employee_id: str, work_day: date, session: str) -> tuple[str, str, str]:
    return employee_id, work_day.isoformat(), session


def _csv_bytes(
    employees: list[dict], week_days: list[date], draft: dict[tuple[str, str, str], str]
) -> bytes:
    rows = []
    for employee in employees:
        row = {
            "Mã NV": employee["employee_id"],
            "Nhân viên": employee["full_name"],
            "Phòng ban": employee["department"],
        }
        for day_name, work_day in zip(DAY_NAMES, week_days):
            for session_code, session_label in WORK_SESSIONS:
                row[f"{day_name} {work_day:%d/%m} - {session_label}"] = draft.get(
                    _draft_key(employee["employee_id"], work_day, session_code), UNSCHEDULED
                )
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def render(db: Database, role: str) -> None:
    require_permission(role, "schedule.view")
    today = date.today()
    st.session_state.setdefault("schedule_week_start", _monday(today))
    week_start: date = st.session_state.schedule_week_start
    week_days = [week_start + timedelta(days=offset) for offset in range(5)]
    week_end = week_days[-1]
    employees = db.list_employees()
    schedules = db.list_work_schedules(week_start.isoformat(), week_end.isoformat())
    owner = st.session_state.get("employee_id")
    if role == EMPLOYEE:
        employees = [row for row in employees if owner and row["employee_id"] == owner]
        schedules = [row for row in schedules if owner and row["employee_id"] == owner]

    drafts = st.session_state.setdefault("work_schedule_drafts", {})
    week_id = week_start.isoformat() + (":" + str(owner) if role == EMPLOYEE else ":ADMIN")
    if week_id not in drafts:
        drafts[week_id] = {
            (row["employee_id"], row["work_date"], row["work_session"]): row["work_status"]
            for row in schedules
        }
    draft: dict[tuple[str, str, str], str] = drafts[week_id]

    st.markdown(
        '<div class="ws-page-head">'
        '<div><div class="ws-title-line">'
        '<h1>Đăng ký Lịch Làm việc Tuần</h1>'
        '<span class="ws-standard"><span></span>Tuần chuẩn hoá</span>'
        '</div><p>Đăng ký và theo dõi trạng thái làm việc tại văn phòng (On-site), WFH hoặc nghỉ phép.</p></div>'
        '<div class="ws-view-switch"><span class="active">▦&nbsp; Bảng tổng hợp</span>'
        '<span>▣&nbsp; Theo ca</span></div></div>',
        unsafe_allow_html=True,
    )
    if notice := st.session_state.pop("schedule_save_notice", None):
        st.success(notice)

    registered_ids = {key[0] for key in draft}
    total_slots = len(employees) * len(week_days) * len(WORK_SESSIONS)
    on_count = sum(value == "ON" for value in draft.values())
    scheduled_count = len(draft)
    on_ratio = round(on_count / scheduled_count * 100) if scheduled_count else 0
    today_keys = [key for key in draft if key[1] == today.isoformat()]
    today_wfh = sum(draft[key] == "WFH" for key in today_keys)
    today_off = sum(draft[key] == "OFF" for key in today_keys)
    department_count = len({row["department"] for row in employees if row["department"]})
    if role != EMPLOYEE:
        _summary_cards(
            len(employees), department_count, len(registered_ids), on_ratio, today_wfh, today_off, today
        )

    with st.container(key="ws_toolbar"):
        navigation, search_col, department_col = st.columns([2.2, 1.5, 2.2], gap="small")
        with navigation:
            previous, week_label, following = st.columns([0.35, 2.7, 0.35], gap="small")
            if previous.button("", icon=":material/chevron_left:", key="ws_previous", help="Tuần trước"):
                _set_week(week_start - timedelta(days=7))
                st.rerun()
            week_label.markdown(
                f'<div class="ws-week-label"><i class="bi bi-calendar3"></i>'
                f'Tuần: {week_start:%d/%m/%Y} – {week_end:%d/%m/%Y}</div>',
                unsafe_allow_html=True,
            )
            if following.button("", icon=":material/chevron_right:", key="ws_next", help="Tuần sau"):
                _set_week(week_start + timedelta(days=7))
                st.rerun()
            if st.button("Tuần này", key="ws_current", type="tertiary"):
                _set_week(today)
                st.rerun()
        query = search_col.text_input(
            "Tìm nhân viên",
            placeholder="Tìm theo tên hoặc Mã NV...",
            icon=":material/search:",
            label_visibility="collapsed",
        ).strip().casefold()
        departments = sorted({row["department"] for row in employees if row["department"]})
        selected_department = department_col.pills(
            "Phòng ban",
            ["Tất cả", *departments],
            default="Tất cả",
            key="ws_department",
            label_visibility="collapsed",
            width="stretch",
        ) or "Tất cả"
        st.markdown(
            '<div class="ws-legend">'
            '<span class="ws-legend-title"><i class="bi bi-info-circle-fill"></i> Chú thích trạng thái:</span>'
            '<span class="ws-legend-on"><i class="bi bi-building-fill"></i><b>ON</b> (Làm tại văn phòng)</span>'
            '<span class="ws-legend-wfh"><i class="bi bi-house-door-fill"></i><b>WFH</b> (Làm từ xa)</span>'
            '<span class="ws-legend-off"><i class="bi bi-cup-hot-fill"></i><b>OFF</b> (Nghỉ phép)</span>'
            '<span class="ws-legend-empty">Chưa đăng ký</span>'
            f'<span class="ws-hours"><i class="bi bi-clock"></i> Khung giờ: <b>Sáng</b> '
            f'({settings.work_start_time:%H:%M} - {settings.morning_end_time:%H:%M}) • '
            f'<b>Chiều</b> ({settings.afternoon_start_time:%H:%M} - {settings.work_end_time:%H:%M})</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    filtered = [
        employee for employee in employees
        if (selected_department == "Tất cả" or employee["department"] == selected_department)
        and (
            not query
            or query in employee["full_name"].casefold()
            or query in employee["employee_id"].casefold()
        )
    ]

    if not employees:
        st.info("Chưa có nhân viên để đăng ký lịch làm việc.")
        return

    with st.container(key="ws_table"):
        header_columns = st.columns([2.7, 1, 1, 1, 1, 1], gap="small")
        header_columns[0].markdown(
            '<div class="ws-employee-header">Nhân viên &amp; Phòng ban '
            '<span>Mã NV</span></div>', unsafe_allow_html=True
        )
        for column, day_name, work_day in zip(header_columns[1:], DAY_NAMES, week_days):
            today_badge = '<em>Hôm nay</em>' if work_day == today else ""
            column.markdown(
                f'<div class="ws-day-head"><b>{day_name}</b>{today_badge}'
                f'<small>{work_day:%d/%m}</small><span>Sáng</span><span>Chiều</span></div>',
                unsafe_allow_html=True,
            )

        if not filtered:
            st.info("Không có nhân viên phù hợp với bộ lọc hiện tại.")
        for row_index, employee in enumerate(filtered):
            with st.container(key=f'ws_row_{employee["employee_id"]}'):
                columns = st.columns([2.7, 1, 1, 1, 1, 1], gap="small")
                color = AVATAR_COLORS[row_index % len(AVATAR_COLORS)]
                columns[0].markdown(
                    f'<div class="ws-employee"><span class="ws-avatar" style="background:{color}">'
                    f'{escape(_initials(employee["full_name"]))}</span><span class="ws-person">'
                    f'<b>{escape(employee["full_name"])}</b><small>{escape(employee["position"] or "Nhân viên")}</small>'
                    f'</span><span class="ws-dept">{escape(employee["department"] or "—")}</span>'
                    f'<code>{escape(employee["employee_id"])}</code></div>',
                    unsafe_allow_html=True,
                )
                for day_column, work_day in zip(columns[1:], week_days):
                    morning, afternoon = day_column.columns(2, gap="small")
                    for cell, (session_code, session_label) in zip(
                        (morning, afternoon), WORK_SESSIONS
                    ):
                        _status_cell(cell, draft, employee["employee_id"], work_day,
                                     session_code, session_label)

        if role != EMPLOYEE:
            footer_columns = st.columns([2.7, 1, 1, 1, 1, 1], gap="small")
            footer_columns[0].markdown(
                '<div class="ws-total-label"><b>Tổng hợp On-site / Buổi</b></div>',
                unsafe_allow_html=True,
            )
            for column, work_day in zip(footer_columns[1:], week_days):
                morning_on = sum(
                    draft.get(_draft_key(employee["employee_id"], work_day, "MORNING")) == "ON"
                    for employee in employees
                )
                afternoon_on = sum(
                    draft.get(_draft_key(employee["employee_id"], work_day, "AFTERNOON")) == "ON"
                    for employee in employees
                )
                column.markdown(
                    f'<div class="ws-day-total"><span><b>{morning_on}</b> ON</span>'
                    f'<span><b>{afternoon_on}</b> ON</span></div>', unsafe_allow_html=True
                )

        action_info, apply_on, friday_wfh, save = st.columns([3.5, 1.2, 1, 1], gap="small")
        action_info.markdown(
            f'<div class="ws-table-info">Hiển thị <b>{len(filtered)}</b> trên tổng số '
            f'<b>{len(employees)}</b> nhân sự</div>', unsafe_allow_html=True
        )
        if apply_on.button("Áp dụng cả tuần ON", key="ws_apply_on", width="stretch"):
            for employee in filtered:
                for work_day in week_days:
                    for session_code, _ in WORK_SESSIONS:
                        draft[_draft_key(employee["employee_id"], work_day, session_code)] = "ON"
            st.rerun()
        if friday_wfh.button("WFH thứ 6", key="ws_friday_wfh", width="stretch"):
            for employee in filtered:
                for session_code, _ in WORK_SESSIONS:
                    draft[_draft_key(employee["employee_id"], week_days[-1], session_code)] = "WFH"
            st.rerun()
        if save.button("Lưu thay đổi", key="ws_save", type="primary", width="stretch"):
            require_permission(role, "schedule.update")
            entries = []
            for employee in employees:
                for work_day in week_days:
                    for session_code, _ in WORK_SESSIONS:
                        value = draft.get(_draft_key(employee["employee_id"], work_day, session_code))
                        entries.append((employee["employee_id"], work_day.isoformat(), session_code, value))
            db.save_work_schedules(entries, actor_role=role, actor_employee_id=owner)
            st.session_state.schedule_save_notice = (
                f"Đã lưu lịch tuần {week_start:%d/%m} – {week_end:%d/%m/%Y}."
            )
            st.rerun()

    export_col, tip_col = st.columns([1, 4])
    export_col.download_button(
        "Xuất Excel / CSV",
        data=_csv_bytes(employees, week_days, draft),
        file_name=f"lich-lam-viec-{week_id}.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )
    tip_col.markdown(
        '<div class="ws-tip"><i class="bi bi-lightbulb-fill"></i><span>'
        '<b>Lưu ý lịch làm việc</b><small>ON cho phép camera chấm công; WFH, OFF và chưa đăng ký '
        'không tạo attendance. Giờ nghỉ trưa không được tính là đi muộn, về sớm hoặc vắng.</small>'
        '</span></div>', unsafe_allow_html=True
    )


def _status_cell(
    cell,
    draft: dict[tuple[str, str, str], str],
    employee_id: str,
    work_day: date,
    session_code: str,
    session_label: str,
) -> None:
    key = _draft_key(employee_id, work_day, session_code)
    status = draft.get(key, UNSCHEDULED)
    label, icon = STATUS_META.get(status, (f"+ {session_label}", ":material/add:"))
    safe_status = status.lower() if status != UNSCHEDULED else "empty"
    widget_id = f"{employee_id}_{work_day:%Y%m%d}_{session_code.lower()}"
    with cell.popover(
        label,
        icon=icon,
        key=f"ws_status_{safe_status}_{widget_id}",
        width="stretch",
    ):
        selected = st.radio(
            f"{session_label} · {work_day:%d/%m}",
            STATUS_OPTIONS,
            index=STATUS_OPTIONS.index(status),
            key=f"ws_choice_{widget_id}",
        )
        normalized = None if selected == UNSCHEDULED else selected
        current = draft.get(key)
        if normalized != current:
            if normalized is None:
                draft.pop(key, None)
            else:
                draft[key] = normalized
            st.rerun()


def _summary_cards(
    total: int,
    departments: int,
    registered: int,
    on_ratio: int,
    today_wfh: int,
    today_off: int,
    today: date,
) -> None:
    percent = round(registered / total * 100) if total else 0
    remaining = max(0, total - registered)
    st.markdown(
        '<div class="ws-kpis">'
        f'<div class="ws-kpi"><span><small>Tổng nhân sự công ty</small><b>{total} '
        f'<em>nhân viên</em></b><i>{departments} phòng ban trực thuộc</i></span>'
        '<strong class="slate"><i class="bi bi-people-fill"></i></strong></div>'
        f'<div class="ws-kpi"><span><small>Đã gửi lịch tuần này</small><b class="green">{registered}/{total} '
        f'<em>{percent}%</em></b><i>Còn {remaining} nhân viên chưa gửi</i></span>'
        '<strong class="green-bg"><i class="bi bi-check-circle-fill"></i></strong></div>'
        f'<div class="ws-kpi"><span><small>Tỉ lệ On-site TB</small><b class="indigo">{on_ratio}% '
        '<em>công suất</em></b><i>Theo các ca đã đăng ký</i></span>'
        '<strong class="indigo-bg"><i class="bi bi-building-fill"></i></strong></div>'
        f'<div class="ws-kpi"><span><small>Làm từ xa &amp; Nghỉ phép</small><b class="split">'
        f'<u>{today_wfh} WFH</u><em>|</em><mark>{today_off} Nghỉ</mark></b>'
        f'<i>{today:%d/%m} hôm nay</i></span>'
        '<strong class="blue-bg"><i class="bi bi-house-door-fill"></i></strong></div>'
        '</div>', unsafe_allow_html=True
    )
