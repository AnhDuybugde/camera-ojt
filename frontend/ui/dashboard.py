from __future__ import annotations

from datetime import date, timedelta
from html import escape

import altair as alt
import cv2
import pandas as pd
import streamlit as st

from attendance.daily_service import daily_attendance_counts
from attendance.schedule_policy import get_next_week_range, get_week_schedule_completion
from config import settings
from database.db import Database
from ui.components import (
    TableColumn, data_table, employee_cell, kpi_card, page_header, section_header,
    status_badge, vietnamese_date,
)


def render(db: Database) -> None:
    today = date.today()
    today_iso = today.isoformat()
    employees = db.list_employees()
    records = db.list_attendance(today_iso, today_iso)
    schedules = db.list_work_schedules(today_iso, today_iso)
    week_records = db.list_attendance(
        (today - timedelta(days=6)).isoformat(), today_iso
    )
    scheduled_on_ids = {
        row["employee_id"] for row in schedules if row["work_status"] == "ON"
    }
    wfh_ids = {
        row["employee_id"] for row in schedules
        if row["work_status"] == "WFH" and row["employee_id"] not in scheduled_on_ids
    }
    counts = daily_attendance_counts(records, scheduled_on_ids)
    absent_ids = {
        row["employee_id"] for row in records
        if row["employee_id"] in scheduled_on_ids
        and row["status"] == "ABSENT"
    }
    total, present = len(employees), counts["present"]
    late, absent = counts["late"], counts["absent"]
    engines = st.session_state.get("live_engines", {})
    connected = sum(bool(engine.camera.connected) for engine in engines.values())

    page_header(
        "Tổng quan",
        f"{vietnamese_date(today)} · Dữ liệu vận hành theo thời gian thực",
        status=f"{connected}/{len(settings.camera_sources)} camera trực tuyến",
    )

    next_start, next_end = get_next_week_range(today)
    next_rows = db.list_work_schedules(next_start.isoformat(), next_end.isoformat())
    incomplete = [
        employee for employee in employees
        if get_week_schedule_completion(next_rows, employee["employee_id"], next_start).status != "COMPLETE"
    ]
    completed = len(employees) - len(incomplete)
    if incomplete:
        st.warning(
            f"Lịch tuần sau ({next_start:%d/%m} – {next_end:%d/%m/%Y}): "
            f"{completed}/{len(employees)} nhân viên đã hoàn tất; "
            f"{len(incomplete)} nhân viên chưa hoàn tất."
        )
        with st.expander("Xem nhân viên chưa hoàn tất lịch"):
            for employee in incomplete:
                completion = get_week_schedule_completion(next_rows, employee["employee_id"], next_start)
                st.write(f"{employee['employee_id']} · {employee['full_name']} — {completion.registered_days}/5 ngày")
    elif employees:
        st.success(f"Lịch tuần sau: {completed}/{len(employees)} nhân viên đã hoàn tất.")

    # ── KPI cards ────────────────────────────────────────────────────────────
    # Stitch card colors exactly:
    # Card1: primary icon (groups), value on-surface
    # Card2: secondary icon (check_circle), value on-surface + sub secondary
    # Card3: tertiary icon (alarm_on), value tertiary color
    # Card4: error icon (person_off), value error color
    cols = st.columns(5)
    with cols[0]:
        kpi_card("Tổng nhân viên", total, "Hồ sơ đang quản lý",
                 "groups", "#004ac6", "#dce9ff")
    with cols[1]:
        kpi_card("Có mặt hôm nay", present,
                 f"{_ratio(present, len(scheduled_on_ids)):.0f}% lịch ON đã check-in",
                 "check_circle", "#006c49", "rgba(108,248,187,0.30)")
    with cols[2]:
        kpi_card("WFH hôm nay", len(wfh_ids), "Không chấm công tại camera",
                 "home_work", "#004ac6", "#dce9ff")
    with cols[3]:
        kpi_card("Đi muộn", late, "Sau mốc giờ quy định",
                 "alarm_on", "#784b00", "#ffddb8")
    with cols[4]:
        kpi_card("Vắng mặt", absent, "Lịch ON chưa check-in",
                 "person_off", "#ba1a1a", "#ffdad6")

    st.write("")

    # ── Main 2-column: camera (7/12) + chart (5/12) ──────────────────────────
    cam_col, chart_col = st.columns([7, 5], gap="large")

    with cam_col:
        # Camera card – Stitch style
        st.markdown(
            '<div class="stitch-card" style="padding:var(--sp-md);">'
            '  <div class="stitch-camera-header">'
            '    <div>'
            '      <div class="stitch-camera-title">'
            '        <span class="stitch-camera-pulse"></span>'
            '        Camera nhận diện trực tiếp'
            '      </div>'
            '      <div class="stitch-camera-sub">Thiết bị: IMOU · Camera điểm danh</div>'
            '    </div>'
            '    <div class="stitch-camera-badges">'
            '      <span class="stitch-live-badge">'
            '        <span class="stitch-live-dot"></span>LIVE FEED'
            '      </span>'
            '    </div>'
            '  </div>',
            unsafe_allow_html=True,
        )
        if engines:
            _dashboard_camera(engines)
        else:
            st.markdown(
                '<div class="stitch-empty" style="margin:0;">'
                '  <div class="stitch-empty-icon">'
                '    <i class="bi bi-camera-video-off-fill" aria-hidden="true"></i>'
                '  </div>'
                '  <h3>Camera chưa được khởi động</h3>'
                '  <p>Mở màn hình Điểm danh trực tiếp để bắt đầu nhận diện.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.button("Mở Camera trực tiếp", type="primary",
                      icon=":material/videocam:", on_click=_go_live)
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col:
        st.markdown(
            '<div class="stitch-card" style="padding:var(--sp-md);">'
            '  <div class="stitch-section-head">'
            '    <div class="stitch-section-title">Thống kê chấm công tuần</div>'
            '  </div>',
            unsafe_allow_html=True,
        )
        trend = _daily_trend(week_records, today)
        melted = trend.melt("Ngày", var_name="Nhóm", value_name="Số nhân viên")
        # Stitch chart colors: primary=#004ac6 for present, tertiary-container=#996100 for late
        chart = (
            alt.Chart(melted)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("yearmonthdate(Ngày):O", title=None,
                         axis=alt.Axis(format="%d/%m", labelAngle=0,
                                       labelFontSize=11, labelColor="#434655")),
                y=alt.Y("Số nhân viên:Q", title=None,
                         axis=alt.Axis(labelFontSize=11, labelColor="#434655")),
                color=alt.Color(
                    "Nhóm:N", title=None,
                    scale=alt.Scale(
                        domain=["Đúng giờ", "Đi muộn"],
                        range=["#004ac6", "#996100"],
                    ),
                    legend=alt.Legend(orient="top", titleFontSize=11, labelFontSize=11),
                ),
                tooltip=[
                    alt.Tooltip("Ngày:T", format="%d/%m/%Y"),
                    "Nhóm:N",
                    "Số nhân viên:Q",
                ],
            )
            .properties(height=260, background="transparent")
        )
        st.altair_chart(chart, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # ── Recent attendance ────────────────────────────────────────────────────
    recent = db.list_attendance()[:8]
    section_header("Chấm công gần đây", "Dữ liệu ghi nhận thời gian thực")
    if recent:
        data_table(
            [
                TableColumn("employee", "Nhân viên", 320),
                TableColumn("department", "Phòng ban", 160),
                TableColumn("date", "Ngày", 120),
                TableColumn("check_in", "Giờ vào", 110),
                TableColumn("check_out", "Giờ ra", 110),
                TableColumn("status", "Trạng thái", 150),
            ],
            [{
                "employee": employee_cell(
                    row["employee_name"], row["employee_id"],
                    row["department"] or "Nhân viên",
                ),
                "department": row["department"] or "—",
                "date": row["date"],
                "check_in": _clock(row["check_in"]),
                "check_out": _clock(row["check_out"]),
                "status": status_badge(
                    {"ON_TIME": "Đúng giờ", "LATE": "Đi muộn", "WAITING": "Chờ chấm công",
                     "ABSENT": "Vắng mặt"}.get(row["status"], row["status"]),
                    {"ON_TIME": "success", "LATE": "warning", "WAITING": "neutral",
                     "ABSENT": "danger"}.get(row["status"], "neutral"),
                ),
            } for row in recent],
            compact=True,
            max_height=360,
        )
    else:
        data_table([], [], empty_message="Chưa có dữ liệu điểm danh.")

    st.write("")

    # ── System status + missing employees ─────────────────────────────────────
    status_col, missing_col = st.columns([1, 1.7], gap="large")
    with status_col:
        section_header("Trạng thái hệ thống", "Camera và đồng bộ")
        with st.container(border=True):
            _system_status(db)
    with missing_col:
        section_header("Nhân sự vắng mặt", "Chỉ gồm lịch ON chưa check-in hôm nay")
        with st.container(border=True):
            missing = [row for row in employees if row["employee_id"] in absent_ids]
            if missing:
                data_table(
                    [
                        TableColumn("employee", "Nhân viên", 220),
                        TableColumn("department", "Phòng ban", 120),
                        TableColumn("status", "Trạng thái", 95),
                    ],
                    [{
                        "employee": employee_cell(
                            row["full_name"], row["employee_id"], row["department"] or "Nhân viên"
                        ),
                        "department": row["department"] or "—",
                        "status": status_badge("Vắng", "danger"),
                    } for row in missing],
                    compact=True,
                )
            else:
                st.success("Không có nhân viên lịch ON bị vắng.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _go_live() -> None:
    st.session_state.active_page = "Điểm danh trực tiếp"


@st.fragment(run_every=0.2)
def _dashboard_camera(engines: dict) -> None:
    name, engine = next(iter(engines.items()))
    connected = engine.camera.connected
    frame, detections = engine.latest()

    conn_label = "Đang kết nối" if connected else "Mất kết nối"
    conn_icon = "bi-link-45deg" if connected else "bi-link-slash"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
        f'  <span class="stitch-conn-badge">'
        f'    <i class="bi {conn_icon}" aria-hidden="true"></i> {conn_label}'
        f'  </span>'
        f'  <span style="font-size:11px;color:var(--st-on-surface-variant);">'
        f'  IMOU · {escape(name)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if frame is not None:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                 channels="RGB", width="stretch")
    elif engine.error:
        st.warning(engine.error)
    else:
        st.info("Đang chờ khung hình đầu tiên…")

    age = engine.camera.frame_age_ms
    st.markdown(
        f'<div class="stitch-camera-overlay">'
        f'  <span>{len(detections)} khuôn mặt nhận diện</span>'
        f'  <span>Độ trễ: {age if age is not None else "—"} ms</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _system_status(db: Database) -> None:
    engines = st.session_state.get("live_engines", {})
    for name, _source in settings.camera_sources:
        engine = engines.get(name)
        if engine and engine.camera.connected:
            state, css, sub = "Trực tuyến", "stitch-pill-ok", "Đang nhận diện"
        elif engine:
            state, css, sub = "Kết nối lại", "stitch-pill-absent", "RTSP tạm gián đoạn"
        else:
            state, css, sub = "Sẵn sàng", "stitch-pill-neutral", "Chưa khởi động"
        st.markdown(_status_html(name, sub, state, css), unsafe_allow_html=True)

    pending = len(db.pending_attendance())
    if settings.google_sheet_id and settings.google_credentials_path:
        state, css = (
            ("Đã đồng bộ", "stitch-pill-ok") if pending == 0
            else (f"Chờ {pending}", "stitch-pill-late")
        )
        sub = "Google Sheets"
    else:
        state, css, sub = "Chưa cấu hình", "stitch-pill-neutral", "SQLite vẫn hoạt động"
    st.markdown(_status_html("Đồng bộ dữ liệu", sub, state, css), unsafe_allow_html=True)


def _status_html(name: str, sub: str, state: str, css: str) -> str:
    return (
        f'<div class="stitch-status-panel">'
        f'  <div class="stitch-status-row">'
        f'    <div>'
        f'      <div class="stitch-status-name">{escape(name)}</div>'
        f'      <div class="stitch-status-sub">{escape(sub)}</div>'
        f'    </div>'
        f'    <span class="stitch-pill {css}">'
        f'      <span class="stitch-pill-dot"></span>{escape(state)}'
        f'    </span>'
        f'  </div>'
        f'</div>'
    )


def _daily_trend(records: list[dict], today: date) -> pd.DataFrame:
    days = pd.date_range(today - timedelta(days=6), today, freq="D")
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame({"Ngày": days, "Đúng giờ": 0, "Đi muộn": 0})
    frame["Ngày"] = pd.to_datetime(frame["date"])
    grouped = (
        frame.groupby("Ngày")
        .agg(**{
            "Có mặt": ("employee_id", "nunique"),
            "Đi muộn": ("status", lambda v: (v == "LATE").sum()),
        })
        .reindex(days, fill_value=0)
        .reset_index()
        .rename(columns={"index": "Ngày"})
    )
    grouped["Đúng giờ"] = grouped["Có mặt"] - grouped["Đi muộn"]
    return grouped[["Ngày", "Đúng giờ", "Đi muộn"]]


def _clock(value: str | None) -> str:
    return value.split("T", 1)[-1][:8] if value else "—"


def _ratio(value: int, total: int) -> float:
    return value / total * 100 if total else 0.0
