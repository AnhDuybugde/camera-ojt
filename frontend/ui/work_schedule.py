"""Compact weekly grid shared by administrators and employees."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from auth.permissions import EMPLOYEE, require_permission
from attendance.schedule_policy import (
    can_employee_edit_schedule, get_next_week_range, get_schedule_deadline,
    get_week_schedule_completion,
)
from config import settings
from database.db import Database

UNSCHEDULED = "Chưa đăng ký"
DAY_NAMES = ("Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6")
WORK_SESSIONS = (("MORNING", "Sáng"), ("AFTERNOON", "Chiều"))
STATUS_OPTIONS = ("ON", "WFH", "OFF", UNSCHEDULED)
AVATAR_COLORS = ("#4f46e5", "#2563eb", "#059669", "#db2777", "#475569")
STYLE_PATH = Path(__file__).resolve().parents[1] / "assets" / "schedule.css"


def _monday(value: date) -> date:
    return value - timedelta(days=value.weekday())


def _clear_selection() -> None:
    for key in st.session_state:
        if key.startswith("sg_select_"):
            st.session_state[key] = False


def _set_week(value: date) -> None:
    st.session_state.schedule_week_start = _monday(value)
    if "sg_calendar_date" in st.session_state:
        st.session_state.sg_calendar_date = value
    _clear_selection()


def _select_calendar_date() -> None:
    _set_week(st.session_state.sg_calendar_date)


def _initials(name: str) -> str:
    return "".join(part[0].upper() for part in name.split()[-2:]) or "NV"


def _draft_key(employee_id: str, work_day: date, session: str) -> tuple[str, str, str]:
    return employee_id, work_day.isoformat(), session


def _csv_bytes(employees: list[dict], week_days: list[date], schedule: dict) -> bytes:
    rows = []
    for employee in employees:
        row = {"Mã NV": employee["employee_id"], "Nhân viên": employee["full_name"],
               "Phòng ban": employee["department"]}
        for day_name, work_day in zip(DAY_NAMES, week_days):
            for session_code, session_label in WORK_SESSIONS:
                row[f"{day_name} {work_day:%d/%m} - {session_label}"] = schedule.get(
                    _draft_key(employee["employee_id"], work_day, session_code), UNSCHEDULED)
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def _selection_key(employee_id: str) -> str:
    return f"sg_select_{employee_id}"


def _select_visible(employee_ids: list[str]) -> None:
    for employee_id in employee_ids:
        st.session_state[_selection_key(employee_id)] = st.session_state.sg_select_all


def _editor_keys(token: str) -> tuple[str, str]:
    return f"sg_morning_{token}", f"sg_afternoon_{token}"


def _open_editor(popover_key: str, token: str, morning: str, afternoon: str) -> None:
    if not st.session_state.get(popover_key):
        return
    previous = st.session_state.get("sg_open_editor")
    if previous and previous != popover_key:
        st.session_state[previous] = False
    st.session_state.sg_open_editor = popover_key
    for key, value in zip(_editor_keys(token), (morning, afternoon)):
        st.session_state[key] = value
    st.session_state.pop(f"sg_error_{token}", None)


def _set_full_day(token: str, status: str) -> None:
    for key in _editor_keys(token):
        st.session_state[key] = status


def _close_editor(popover_key: str) -> None:
    st.session_state[popover_key] = False


def _save_day(db: Database, employee_id: str, work_day: date, role: str,
              owner: str | None, token: str, popover_key: str) -> None:
    # Save this day only: other days and other employees are never replayed from drafts.
    entries = []
    for (session, _), key in zip(WORK_SESSIONS, _editor_keys(token)):
        value = st.session_state.get(key)
        if value not in STATUS_OPTIONS:
            st.session_state[f"sg_error_{token}"] = "Vui lòng chọn trạng thái cho cả sáng và chiều."
            return
        entries.append((employee_id, work_day.isoformat(), session,
                        None if value == UNSCHEDULED else value))
    try:
        db.save_work_schedules(entries, actor_role=role, actor_employee_id=owner)
    except (PermissionError, ValueError) as exc:
        st.session_state[f"sg_error_{token}"] = str(exc)
    else:
        _close_editor(popover_key)
        st.session_state.sg_saved_notice = f"Đã lưu lịch ngày {work_day:%d/%m}."


def _apply_week(db: Database, employee_ids: list[str], week_days: list[date],
                role: str, owner: str | None) -> None:
    status = st.session_state.get("sg_batch_status", "ON")
    entries = [(employee_id, day.isoformat(), session, status)
               for employee_id in employee_ids for day in week_days
               for session, _ in WORK_SESSIONS]
    try:
        db.save_work_schedules(entries, actor_role=role, actor_employee_id=owner)
    except (PermissionError, ValueError) as exc:
        st.session_state.sg_batch_error = str(exc)
    else:
        st.session_state.sg_saved_notice = f"Đã cập nhật cả tuần cho {len(employee_ids)} nhân viên."
        st.session_state.pop("sg_batch_error", None)
        _clear_selection()


def _day_label(morning: str, afternoon: str) -> str:
    if morning == afternoon:
        return morning if morning == UNSCHEDULED else f"**{morning}**"
    short = lambda value: "—" if value == UNSCHEDULED else value
    return f"Sáng: **{short(morning)}**  \nChiều: **{short(afternoon)}**"


def _day_cell(db: Database, employee: dict, work_day: date, schedule: dict,
              editable: bool, role: str, owner: str | None) -> None:
    employee_id = employee["employee_id"]
    morning, afternoon = [schedule.get(_draft_key(employee_id, work_day, session), UNSCHEDULED)
                          for session, _ in WORK_SESSIONS]
    token = f"{role}_{employee_id}_{work_day:%Y%m%d}"
    popover_key = f"sg_edit_{token}"
    tone = lambda value: value.lower() if value != UNSCHEDULED else "empty"
    with st.container(key=f"sg_cell_m_{tone(morning)}_a_{tone(afternoon)}_{token}"):
        popover = st.popover(
            _day_label(morning, afternoon), key=popover_key, width="stretch", wrap=True,
            disabled=not editable,
            help=f"{work_day:%d/%m} · Sáng: {morning} · Chiều: {afternoon}",
            on_change=_open_editor, args=(popover_key, token, morning, afternoon))
        if not popover.open or not editable:
            return
        with popover, st.container(key=f"sg_editor_{token}"):
            st.markdown(
                f'<div class="sg-editor-title">{escape(employee["full_name"])}</div>'
                f'<div class="sg-editor-date">{DAY_NAMES[work_day.weekday()]} · {work_day:%d/%m/%Y}</div>',
                unsafe_allow_html=True)
            st.caption("Chọn nhanh cả ngày")
            quick = st.columns(3, gap="small")
            for col, status in zip(quick, ("ON", "WFH", "OFF")):
                col.button(status, key=f"sg_quick_{status}_{token}", width="stretch",
                           on_click=_set_full_day, args=(token, status))
            labels = (
                f"Sáng · {settings.work_start_time:%H:%M}–{settings.morning_end_time:%H:%M}",
                f"Chiều · {settings.afternoon_start_time:%H:%M}–{settings.work_end_time:%H:%M}")
            for key, label in zip(_editor_keys(token), labels):
                st.segmented_control(label, STATUS_OPTIONS, key=key, required=True,
                                     format_func=lambda x: "Chưa chọn" if x == UNSCHEDULED else x,
                                     width="stretch")
            if error := st.session_state.get(f"sg_error_{token}"):
                st.error(error)
            cancel, save = st.columns(2)
            cancel.button("Hủy", key=f"sg_cancel_{token}", width="stretch",
                          on_click=_close_editor, args=(popover_key,))
            save.button("Lưu ngày", key=f"sg_save_{token}", type="primary", width="stretch",
                        on_click=_save_day, args=(db, employee_id, work_day, role, owner, token, popover_key))


def _progress(schedules: list[dict], employees: list[dict], week_start: date,
              now: datetime, role: str, owner: str | None, editable: bool) -> None:
    if role == EMPLOYEE:
        completion = get_week_schedule_completion(schedules, owner or "", week_start)
        tone = "complete" if completion.status == "COMPLETE" else "pending"
        summary = f"{completion.registered_days}/5 ngày · " + (
            "Đã hoàn tất" if completion.status == "COMPLETE" else "Chưa hoàn tất")
        detail = (f"Hạn đăng ký: Thứ Sáu, {get_schedule_deadline(now):%d/%m} · 23:59"
                  if editable else "Chỉ xem · Chỉ sửa tuần sau từ Thứ Hai đến hết Thứ Sáu")
        if editable and now.weekday() == 4:
            detail = "Hôm nay là hạn cuối · " + detail
    else:
        count = sum(get_week_schedule_completion(schedules, emp["employee_id"], week_start).status
                    == "COMPLETE" for emp in employees)
        tone = "complete" if count == len(employees) else "pending"
        summary = f"{count}/{len(employees)} nhân viên hoàn tất"
        detail = "Đủ sáng & chiều từ Thứ 2 đến Thứ 6"
    st.markdown(f'<div class="sg-progress {tone}"><b>{summary}</b><span>{detail}</span></div>',
                unsafe_allow_html=True)


def render(db: Database, role: str) -> None:
    require_permission(role, "schedule.view")
    st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    now = datetime.now()
    today = now.date()
    next_start, _ = get_next_week_range(now)
    owner = st.session_state.get("employee_id")
    identity = f"{role}:{owner or ''}"
    if st.session_state.get("schedule_view_identity") != identity:
        st.session_state.schedule_view_identity = identity
        _set_week(next_start if role == EMPLOYEE else today)
    week_start = st.session_state.setdefault("schedule_week_start", next_start if role == EMPLOYEE else _monday(today))
    week_days = [week_start + timedelta(days=i) for i in range(5)]
    employees = db.list_employees()
    if role == EMPLOYEE:
        employees = [emp for emp in employees if owner and emp["employee_id"] == owner]
        if not employees:
            st.info("Tài khoản chưa được liên kết với hồ sơ nhân viên.")
            return
    schedules = db.list_work_schedules(week_start.isoformat(), week_days[-1].isoformat(),
                                       employee_id=owner if role == EMPLOYEE else None)
    schedule = {(row["employee_id"], row["work_date"], row["work_session"]): row["work_status"]
                for row in schedules}
    editable = role != EMPLOYEE or bool(owner and can_employee_edit_schedule(week_start, now))
    if notice := st.session_state.pop("sg_saved_notice", None):
        st.toast(notice, icon=":material/check_circle:")

    query = ""
    with st.container(key="sg_toolbar"):
        toolbar = st.columns([1.25, 2.5, 1.5, .85] if role != EMPLOYEE else [1.4, 2.5, .85],
                             gap="small", vertical_alignment="center")
        toolbar[0].markdown('<div class="sg-title">' + ("Lịch của tôi" if role == EMPLOYEE else "Lịch làm việc")
                            + '</div><div class="sg-subtitle">Đăng ký theo tuần</div>', unsafe_allow_html=True)
        with toolbar[1]:
            prev, period, following, calendar = st.columns([.4, 2.25, .4, 1.05], gap="small", vertical_alignment="center")
            prev.button("", icon=":material/chevron_left:", key="sg_prev", help="Tuần trước",
                        on_click=_set_week, args=(week_start - timedelta(days=7),))
            period.markdown(f'<div class="sg-week">{week_start:%d/%m} – {week_days[-1]:%d/%m/%Y}</div>',
                            unsafe_allow_html=True)
            following.button("", icon=":material/chevron_right:", key="sg_next", help="Tuần sau",
                             on_click=_set_week, args=(week_start + timedelta(days=7),))
            calendar.date_input(
                "Chọn ngày", value=week_start, format="DD/MM/YYYY",
                key="sg_calendar_date", label_visibility="collapsed",
                on_change=_select_calendar_date,
                help="Chọn một ngày để mở tuần chứa ngày đó",
            )
        if role != EMPLOYEE:
            query = toolbar[2].text_input("Tìm nhân viên", key="sg_search", placeholder="Tên hoặc mã NV…",
                                          icon=":material/search:", label_visibility="collapsed").strip().casefold()
        toolbar[-1].download_button("Xuất CSV", data=_csv_bytes(employees, week_days, schedule),
                                    file_name=f"lich-lam-viec-{week_start}.csv", mime="text/csv",
                                    icon=":material/download:", width="stretch")

    department = "Tất cả"
    registration_status = "ALL"
    completions = {
        emp["employee_id"]: get_week_schedule_completion(schedules, emp["employee_id"], week_start)
        for emp in employees
    }
    with st.container(key="sg_context"):
        if role != EMPLOYEE:
            filters, progress = st.columns([1.2, 1.5], vertical_alignment="center")
            department = filters.pills("Phòng ban", ["Tất cả", *sorted({e["department"] for e in employees if e["department"]})],
                                       default="Tất cả", key="sg_department", label_visibility="collapsed") or "Tất cả"
            with progress:
                _progress(schedules, employees, week_start, now, role, owner, editable)
            complete_count = sum(item.status == "COMPLETE" for item in completions.values())
            incomplete_count = sum(item.status == "INCOMPLETE" for item in completions.values())
            unregistered_count = sum(item.status == "UNREGISTERED" for item in completions.values())
            status_labels = {
                "ALL": f"Tất cả ({len(employees)})",
                "COMPLETE": f"Đã hoàn thành ({complete_count})",
                "INCOMPLETE": f"Chưa hoàn thành ({incomplete_count})",
                "UNREGISTERED": f"Chưa đăng ký ({unregistered_count})",
            }
            registration_status = st.segmented_control(
                "Tiến độ đăng ký", tuple(status_labels), default="ALL",
                format_func=status_labels.__getitem__, key="sg_registration_filter",
                label_visibility="collapsed", width="stretch",
            )
        else:
            _progress(schedules, employees, week_start, now, role, owner, editable)
    filtered = [emp for emp in employees
                if (department == "Tất cả" or emp["department"] == department)
                and (not query or query in emp["full_name"].casefold() or query in emp["employee_id"].casefold())]
    if registration_status != "ALL":
        filtered = [emp for emp in filtered if completions[emp["employee_id"]].status == registration_status]
    view_signature = (identity, week_start, query, department, registration_status)
    if st.session_state.get("sg_selection_view") != view_signature:
        _clear_selection()
        st.session_state.sg_selection_view = view_signature
    if not filtered:
        st.info("Không có nhân viên phù hợp." if employees else "Chưa có nhân viên để đăng ký lịch.")
        return

    widths = [2.55, *([1] * 5)]
    with st.container(key="sg_grid"):
        with st.container(key="sg_header"):
            columns = st.columns(widths, gap="small", wrap=False)
            with columns[0]:
                if role != EMPLOYEE:
                    check, label = st.columns([.3, 2.25], gap="small", vertical_alignment="center")
                    st.session_state.sg_select_all = all(st.session_state.get(_selection_key(e["employee_id"]), False) for e in filtered)
                    check.checkbox("Chọn tất cả nhân viên đang hiển thị", key="sg_select_all", label_visibility="collapsed",
                                   on_change=_select_visible, args=([e["employee_id"] for e in filtered],))
                    label.markdown('<div class="sg-employee-heading">Nhân viên</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="sg-employee-heading">Lịch cá nhân</div>', unsafe_allow_html=True)
            for col, name, day in zip(columns[1:], DAY_NAMES, week_days):
                cls = "today" if day == today else ""
                badge = '<em>Hôm nay</em>' if day == today else ""
                col.markdown(f'<div class="sg-day-head {cls}"><b>{name}</b><span>{day:%d/%m}</span>{badge}</div>',
                             unsafe_allow_html=True)
        for index, emp in enumerate(filtered):
            with st.container(key=f'sg_row_{emp["employee_id"]}'):
                columns = st.columns(widths, gap="small", wrap=False)
                person = columns[0]
                if role != EMPLOYEE:
                    check, person = columns[0].columns([.3, 2.25], gap="small", vertical_alignment="center")
                    check.checkbox(f'Chọn {emp["full_name"]}', key=_selection_key(emp["employee_id"]), label_visibility="collapsed")
                count = completions[emp["employee_id"]].registered_days
                person.markdown(
                    f'<div class="sg-person"><span class="sg-avatar" style="background:{AVATAR_COLORS[index % len(AVATAR_COLORS)]}">'
                    f'{escape(_initials(emp["full_name"]))}</span><span class="sg-person-text">'
                    f'<b title="{escape(emp["full_name"], quote=True)}">{escape(emp["full_name"])}</b>'
                    f'<small>{escape(emp["department"] or "Nhân viên")} · {escape(emp["employee_id"])}</small></span>'
                    f'<span class="sg-count {"done" if count == 5 else ""}">{count}/5</span></div>', unsafe_allow_html=True)
                for col, day in zip(columns[1:], week_days):
                    with col:
                        _day_cell(db, emp, day, schedule, editable, role, owner)

    selected_ids = [emp["employee_id"] for emp in filtered if st.session_state.get(_selection_key(emp["employee_id"]), False)]
    if role != EMPLOYEE:
        with st.container(key="sg_batch"):
            label, choice, apply, cancel = st.columns([1.6, 1.3, 1, .7], vertical_alignment="center")
            if selected_ids:
                label.markdown(f'**Đã chọn {len(selected_ids)} nhân viên**  \nÁp dụng cả sáng và chiều trong 5 ngày')
            else:
                label.markdown('**Áp dụng lịch hàng loạt**  \nTick nhân viên cần cập nhật lịch')
            choice.segmented_control("Trạng thái cả tuần", ("ON", "WFH", "OFF"), default="ON", required=True,
                                     key="sg_batch_status", label_visibility="collapsed", width="stretch",
                                     disabled=not selected_ids)
            apply.button("Áp dụng 5 ngày", type="primary", key="sg_batch_apply", width="stretch",
                         on_click=_apply_week, args=(db, selected_ids, week_days, role, owner),
                         disabled=not selected_ids)
            cancel.button("Bỏ chọn", key="sg_batch_clear", width="stretch", on_click=_clear_selection,
                          disabled=not selected_ids)
            if error := st.session_state.get("sg_batch_error"):
                st.error(error)
    instruction = "Bấm vào ô để chỉnh sáng / chiều" if editable else "Lịch chỉ xem · Liên hệ quản trị viên nếu cần điều chỉnh"
    st.markdown('<div class="sg-footer"><div class="sg-legend"><span class="on">● ON · Văn phòng</span>'
                '<span class="wfh">● WFH · Từ xa</span><span class="off">● OFF · Nghỉ</span>'
                '<span class="empty">○ Chưa đăng ký</span></div>'
                f'<span>{instruction}</span></div>', unsafe_allow_html=True)
