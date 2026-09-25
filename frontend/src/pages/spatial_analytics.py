"""Business-focused spatial analytics dashboard."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from auth.permissions import ADMIN, PermissionDenied
from camera.camera_manager import CameraManager
from config import settings
from database.db import Database
from face.detector import FaceDetector
from face.recognizer import FaceRecognizer
from spatial.business_analytics import (
    attendance_flow, attendance_frame, filter_spatial_events, gate_flow,
    occupancy_timeline, peak,
)
from spatial.detector import PersonDetector
from spatial.engine import SpatialAnalyticsEngine
from spatial.repository import SpatialRepository, local_now
from spatial.zones import Zone, zones_from_environment
from frontend.src.components.components import (
    TableColumn, data_table, kpi_card, page_header, section_header, status_badge,
)


ROOT = settings.base_dir


@st.cache_resource
def _repository() -> SpatialRepository:
    return SpatialRepository(ROOT / "data" / "spatial_analytics.db")


@st.cache_resource
def _person_detector() -> PersonDetector:
    return PersonDetector(ROOT / "yolo11s.pt")


def render(
    db: Database, role: str, face_detector: FaceDetector, recognizer: FaceRecognizer,
) -> None:
    if role != ADMIN:
        raise PermissionDenied("Chỉ quản trị viên được xem phân tích không gian.")
    # In unified mode camera ownership belongs exclusively to the backend.
    from config import settings
    if settings.camera_ojt_url:
        from integration.camera_ojt import CameraOjtClient
        from integration.backend import RemoteService
        st.subheader("Không gian & Ra vào")
        try:
            status = CameraOjtClient(settings.camera_ojt_url).status()
            st.dataframe(status.get("people", []), use_container_width=True)
            health = RemoteService("operations").diagnostics()
            st.caption(f"Sự kiện cần duyệt: {health['review_events']} · "
                       f"Sự kiện chờ đồng bộ: {health['pending_events']}")
        except (RuntimeError, ValueError) as error:
            st.warning(str(error))
        return
    if "spatial_engines" not in st.session_state:
        st.session_state.spatial_engines = {}

    engines: dict[str, SpatialAnalyticsEngine] = st.session_state.spatial_engines
    zones = zones_from_environment()
    repository = _repository()
    detector = _person_detector()
    all_attendance = db.list_attendance()
    default_start, default_end = _default_dates(all_attendance)

    page_header(
        "Phân tích không gian",
        "Dòng người, hiện diện, mật độ không gian và các trường hợp cần xác minh",
        status=f"{sum(engine.camera.connected for engine in engines.values())}/"
               f"{len(settings.camera_sources)} camera phân tích",
    )

    with st.container(border=True):
        start_col, stop_col, date_col, dept_col, bucket_col = st.columns(
            [1, 1, 1.8, 1.7, 1.1], vertical_alignment="bottom",
        )
        if start_col.button(
            "Bắt đầu phân tích", type="primary", width="stretch",
            disabled=bool(engines), icon=":material/play_arrow:",
        ):
            _start_engines(engines, detector, repository, zones, face_detector, recognizer)
            st.rerun()
        if stop_col.button(
            "Dừng phân tích", width="stretch", disabled=not engines,
            icon=":material/stop:",
        ):
            _stop_engines(engines)
            st.rerun()
        selected_dates = date_col.date_input(
            "Khoảng thời gian", value=(default_start, default_end), max_value=date.today(),
            key="spatial_date_range",
        )
        departments = sorted({row.get("department") or "Chưa phân loại" for row in all_attendance})
        selected_departments = dept_col.multiselect(
            "Phòng ban (attendance)", departments, placeholder="Tất cả phòng ban",
            key="spatial_departments",
        )
        bucket_minutes = bucket_col.selectbox(
            "Khoảng gom", [15, 30, 60], index=0,
            format_func=lambda value: f"{value} phút", key="spatial_bucket",
        )

    start_day, end_day = _date_range(selected_dates, default_start, default_end)
    rows = db.list_attendance(start_day.isoformat(), end_day.isoformat())
    attendance = attendance_frame(rows, selected_departments)
    since = datetime.combine(start_day, time.min, tzinfo=local_now().tzinfo)
    transitions = filter_spatial_events(
        repository.transition_rows(since, limit=5000), start_day, end_day,
    )
    density_rows = _filter_timestamp_rows(
        repository.density_rows(since), "captured_at", start_day, end_day,
    )
    visit_rows = _filter_timestamp_rows(
        repository.visit_rows(since), "entered_at", start_day, end_day,
    )
    flow = attendance_flow(attendance, bucket_minutes)
    physical_flow = gate_flow(transitions, bucket_minutes)
    allowed_employee_ids = {
        row["employee_id"] for row in db.list_employees()
        if not selected_departments
        or (row.get("department") or "Chưa phân loại") in selected_departments
    }
    anomaly_transitions = [
        row for row in transitions
        if not row.get("employee_id") or row.get("employee_id") in allowed_employee_ids
    ]

    _summary_kpis(attendance, flow, engines)
    _insight_strip(attendance, flow, density_rows, zones, engines)

    people_tab, presence_tab, space_tab, anomaly_tab = st.tabs([
        "Dòng người", "Hiện diện", "Không gian", "Bất thường",
    ])
    with people_tab:
        _people_flow_tab(attendance, flow, physical_flow, bucket_minutes)
    with presence_tab:
        _presence_tab(
            db, attendance, density_rows, engines, start_day, end_day,
            selected_departments, bucket_minutes,
        )
    with space_tab:
        _space_tab(engines, zones, density_rows, transitions, visit_rows)
    with anomaly_tab:
        _anomaly_tab(attendance, anomaly_transitions, engines, end_day)


def _summary_kpis(
    attendance: pd.DataFrame, flow: pd.DataFrame,
    engines: dict[str, SpatialAnalyticsEngine],
) -> None:
    check_in_peak, check_in_count = peak(flow, "Check-in")
    check_out_peak, check_out_count = peak(flow, "Check-out")
    today_rows = attendance[attendance["date"].astype(str) == date.today().isoformat()] if not attendance.empty else attendance
    open_attendance = int((today_rows["check_in"].notna() & today_rows["check_out"].isna()).sum()) if not today_rows.empty else 0
    camera_people = sum(sum(engine.current_counts().values()) for engine in engines.values())
    cards = (
        ("Cao điểm check-in", check_in_peak, f"{check_in_count} lượt trong một mốc", "login", "#2563eb", "#dce9ff"),
        ("Cao điểm check-out", check_out_peak, f"{check_out_count} lượt trong một mốc", "event_available", "#8b5cf6", "#ede9fe"),
        ("Hiện diện attendance", open_attendance, "Đã vào · chưa check-out hôm nay", "groups", "#10b981", "#d1fae5"),
        ("Camera đang thấy", camera_people, "Tổng người trong các vùng", "videocam", "#f59e0b", "#fef3c7"),
    )
    for column, card in zip(st.columns(4), cards):
        with column:
            kpi_card(*card)


def _insight_strip(
    attendance: pd.DataFrame, flow: pd.DataFrame, density_rows: list[dict],
    zones: tuple[Zone, ...], engines: dict[str, SpatialAnalyticsEngine],
) -> None:
    in_peak, in_count = peak(flow, "Check-in")
    out_peak, out_count = peak(flow, "Check-out")
    insights = []
    if in_count:
        insights.append(f"Check-in tập trung nhất lúc {in_peak} với {in_count} lượt.")
    if out_count:
        insights.append(f"Check-out tập trung nhất lúc {out_peak} với {out_count} lượt.")
    if density_rows:
        zone_labels = {zone.zone_id: zone.label for zone in zones}
        density = pd.DataFrame(density_rows)
        grouped = density.groupby("zone_id", as_index=False)["people_count"].mean()
        top = grouped.loc[grouped["people_count"].idxmax()]
        insights.append(
            f'{zone_labels.get(top["zone_id"], top["zone_id"])} có mật độ trung bình cao nhất '
            f'({top["people_count"]:.1f} người/mẫu).'
        )
    if engines:
        today_rows = attendance[attendance["date"].astype(str) == date.today().isoformat()] if not attendance.empty else attendance
        attendance_now = int((today_rows["check_in"].notna() & today_rows["check_out"].isna()).sum()) if not today_rows.empty else 0
        camera_now = sum(sum(engine.current_counts().values()) for engine in engines.values())
        if abs(attendance_now - camera_now) >= 2:
            insights.append(
                f"Attendance và camera đang lệch {abs(attendance_now - camera_now)} người; cần kiểm tra phạm vi camera."
            )
    if not insights:
        insights.append("Chưa đủ dữ liệu để tạo nhận xét vận hành trong khoảng đã chọn.")
    st.markdown(
        '<div class="spatial-insights"><b><i class="bi bi-lightbulb-fill"></i> Nhận xét vận hành</b>'
        + "".join(f"<span>{escape(item)}</span>" for item in insights[:4]) + "</div>",
        unsafe_allow_html=True,
    )


def _people_flow_tab(
    attendance: pd.DataFrame, flow: pd.DataFrame,
    physical_flow: pd.DataFrame, bucket_minutes: int,
) -> None:
    section_header(
        "Lưu lượng check-in/check-out",
        f"Tổng lượt theo khung {bucket_minutes} phút; không đánh giá đi muộn hoặc về sớm",
    )
    if flow.empty:
        st.info("Chưa có dữ liệu check-in/check-out trong khoảng đã chọn.")
    else:
        order = flow.sort_values("Phút")["Mốc"].drop_duplicates().tolist()
        chart = (
            alt.Chart(flow)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Mốc:N", sort=order, title="Khung giờ"),
                y=alt.Y("Số lượt:Q", title="Số lượt"),
                color=alt.Color(
                    "Loại:N", scale=alt.Scale(
                        domain=["Check-in", "Check-out"], range=["#2563eb", "#8b5cf6"],
                    ), legend=alt.Legend(orient="top"),
                ),
                xOffset="Loại:N",
                tooltip=["Mốc:N", "Loại:N", "Số lượt:Q"],
            )
            .properties(height=330)
        )
        st.altair_chart(chart, width="stretch")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        section_header("Lưu lượng theo ngày", "Khối lượng vào/ra, không phải đánh giá tuân thủ")
        if attendance.empty:
            st.info("Chưa có dữ liệu.")
        else:
            daily = attendance.groupby("date", as_index=False).agg(
                **{"Check-in": ("check_in", "count"), "Check-out": ("check_out", "count")},
            )
            daily = daily.melt("date", var_name="Loại", value_name="Số lượt")
            daily["Ngày"] = pd.to_datetime(daily["date"])
            lines = (
                alt.Chart(daily)
                .mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X("Ngày:T", title=None), y=alt.Y("Số lượt:Q", title="Số lượt"),
                    color=alt.Color("Loại:N", legend=alt.Legend(orient="top")),
                    tooltip=[alt.Tooltip("Ngày:T", format="%d/%m/%Y"), "Loại:N", "Số lượt:Q"],
                ).properties(height=255)
            )
            st.altair_chart(lines, width="stretch")
    with right:
        section_header("Entry/Exit thực tế", "Suy ra từ chuyển vùng Lối vào ↔ khu vực bên trong")
        if physical_flow.empty:
            st.info("Chưa có sự kiện cắt vùng cửa vào. Cần chạy camera và hiệu chỉnh vùng.")
        else:
            order = physical_flow.sort_values("Phút")["Mốc"].drop_duplicates().tolist()
            physical = (
                alt.Chart(physical_flow)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("Mốc:N", sort=order, title="Khung giờ"),
                    y=alt.Y("Số lượt:Q", title="Số lượt"),
                    color=alt.Color("Loại:N", legend=alt.Legend(orient="top")),
                    xOffset="Loại:N", tooltip=["Mốc:N", "Loại:N", "Số lượt:Q"],
                ).properties(height=255)
            )
            st.altair_chart(physical, width="stretch")


def _presence_tab(
    db: Database, attendance: pd.DataFrame, density_rows: list[dict],
    engines: dict[str, SpatialAnalyticsEngine], start_day: date, end_day: date,
    departments: list[str], bucket_minutes: int,
) -> None:
    selected_day = st.date_input(
        "Ngày xem hiện diện", value=end_day, min_value=start_day, max_value=end_day,
        key="spatial_presence_day",
    )
    day_iso = selected_day.isoformat()
    timeline = occupancy_timeline(attendance, selected_day, bucket_minutes)
    day_rows = attendance[attendance["date"].astype(str) == day_iso] if not attendance.empty else attendance
    checked_in = int(day_rows["check_in"].notna().sum()) if not day_rows.empty else 0
    open_count = int((day_rows["check_in"].notna() & day_rows["check_out"].isna()).sum()) if not day_rows.empty else 0

    schedules = db.list_work_schedules(day_iso, day_iso)
    if departments:
        schedules = [row for row in schedules if (row.get("department") or "Chưa phân loại") in departments]
    morning_on = len({row["employee_id"] for row in schedules if row["work_session"] == "MORNING" and row["work_status"] == "ON"})
    afternoon_on = len({row["employee_id"] for row in schedules if row["work_session"] == "AFTERNOON" and row["work_status"] == "ON"})
    for column, card in zip(st.columns(4), (
        ("Lịch ON buổi sáng", morning_on, selected_day.strftime("%d/%m/%Y"), "calendar_month", "#2563eb", "#dce9ff"),
        ("Lịch ON buổi chiều", afternoon_on, selected_day.strftime("%d/%m/%Y"), "calendar_month", "#8b5cf6", "#ede9fe"),
        ("Đã check-in", checked_in, "Có bản ghi vào trong ngày", "login", "#10b981", "#d1fae5"),
        ("Chưa check-out", open_count, "Hiện diện theo attendance", "groups", "#f59e0b", "#fef3c7"),
    )):
        with column:
            kpi_card(*card)

    section_header("Đường hiện diện trong ngày", "Attendance lũy kế và mật độ camera là hai nguồn độc lập")
    series = []
    if not timeline.empty:
        attendance_line = timeline.rename(columns={"Hiện diện": "Số người"})
        attendance_line["Nguồn"] = "Attendance"
        series.append(attendance_line[["Mốc thời gian", "Số người", "Nguồn"]])
    camera_line = _camera_presence_timeline(density_rows, selected_day, bucket_minutes)
    if not camera_line.empty:
        series.append(camera_line)
    if series:
        combined = pd.concat(series, ignore_index=True)
        line = (
            alt.Chart(combined)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("Mốc thời gian:T", title=None),
                y=alt.Y("Số người:Q", title="Số người", scale=alt.Scale(zero=True)),
                color=alt.Color(
                    "Nguồn:N", scale=alt.Scale(
                        domain=["Attendance", "Camera"], range=["#2563eb", "#10b981"],
                    ), legend=alt.Legend(orient="top"),
                ),
                tooltip=[alt.Tooltip("Mốc thời gian:T", format="%H:%M"), "Nguồn:N", "Số người:Q"],
            ).properties(height=340)
        )
        st.altair_chart(line, width="stretch")
        st.caption(
            "Camera có thể đếm trùng nếu các góc nhìn chồng lấn; đường camera dùng để đối chiếu vận hành, "
            "không thay thế bản ghi chấm công."
        )
    else:
        st.info("Chưa có dữ liệu hiện diện cho ngày đã chọn.")


def _space_tab(
    engines: dict[str, SpatialAnalyticsEngine], zones: tuple[Zone, ...],
    density_rows: list[dict], transitions: list[dict], visits: list[dict],
) -> None:
    _zone_legend(zones)
    if engines:
        section_header("Giám sát trực tiếp", "Track ID, danh tính, vùng và vệt di chuyển")
        _live_fragment(engines, zones)
    else:
        st.info("Nhấn “Bắt đầu phân tích” để thu thập mật độ và luồng vùng từ camera.")
    _space_history(density_rows, transitions, visits, zones)


def _anomaly_tab(
    attendance: pd.DataFrame, transitions: list[dict],
    engines: dict[str, SpatialAnalyticsEngine], end_day: date,
) -> None:
    section_header(
        "Các trường hợp cần xác minh",
        "Đây là tín hiệu đối chiếu dữ liệu, không phải kết luận vi phạm",
    )
    anomalies: list[dict] = []
    attendance_index: dict[tuple[str, str], dict] = {}
    if not attendance.empty:
        for _, row in attendance.iterrows():
            key = (str(row["date"]), str(row["employee_id"]))
            attendance_index[key] = row.to_dict()
            if pd.notna(row.get("check_in")) and pd.isna(row.get("check_out")):
                anomalies.append({
                    "level": "Theo dõi", "type": "Chưa có check-out",
                    "person": f'{row["employee_name"]} ({row["employee_id"]})',
                    "detail": f'Đã check-in ngày {row["date"]} nhưng bản ghi chưa hoàn tất.',
                })

    seen_without_attendance: set[tuple[str, str]] = set()
    after_checkout: set[tuple[str, str]] = set()
    unknown_after_hours = 0
    for event in transitions:
        timestamp = pd.to_datetime(event.get("event_time"), errors="coerce")
        if pd.isna(timestamp):
            continue
        employee_id = event.get("employee_id")
        event_day = timestamp.date().isoformat()
        if employee_id:
            key = (event_day, str(employee_id))
            record = attendance_index.get(key)
            if record is None:
                seen_without_attendance.add(key)
            elif pd.notna(record.get("check_out")):
                checkout = pd.Timestamp(record["check_out"])
                event_time = timestamp.tz_localize(None) if timestamp.tzinfo else timestamp
                if event_time > checkout:
                    after_checkout.add(key)
        elif timestamp.time() < settings.work_start_time or timestamp.time() > settings.work_end_time:
            unknown_after_hours += 1

    for event_day, employee_id in sorted(seen_without_attendance):
        name = next((row.get("employee_name") for row in transitions if row.get("employee_id") == employee_id), employee_id)
        anomalies.append({
            "level": "Cần kiểm tra", "type": "Camera thấy nhưng chưa có attendance",
            "person": f"{name} ({employee_id})", "detail": f"Có sự kiện camera ngày {event_day}.",
        })
    for event_day, employee_id in sorted(after_checkout):
        record = attendance_index[(event_day, employee_id)]
        anomalies.append({
            "level": "Cần kiểm tra", "type": "Xuất hiện sau check-out",
            "person": f'{record["employee_name"]} ({employee_id})',
            "detail": f"Camera ghi nhận chuyển vùng sau thời điểm check-out ngày {event_day}.",
        })
    if unknown_after_hours:
        anomalies.append({
            "level": "An ninh", "type": "Người chưa xác định ngoài giờ",
            "person": "Chưa xác định", "detail": f"{unknown_after_hours} sự kiện chuyển vùng ngoài giờ làm việc.",
        })
    for name, engine in engines.items():
        if not engine.camera.connected:
            anomalies.append({
                "level": "Hệ thống", "type": "Camera mất kết nối", "person": name,
                "detail": engine.error or "Luồng camera đang kết nối lại.",
            })

    if engines and end_day == date.today() and not attendance.empty:
        today_rows = attendance[attendance["date"].astype(str) == date.today().isoformat()]
        attendance_now = int((today_rows["check_in"].notna() & today_rows["check_out"].isna()).sum())
        camera_now = sum(sum(engine.current_counts().values()) for engine in engines.values())
        difference = abs(attendance_now - camera_now)
        if difference >= 2:
            anomalies.append({
                "level": "Đối chiếu", "type": "Chênh lệch số người",
                "person": "Toàn văn phòng",
                "detail": f"Attendance: {attendance_now} · Camera: {camera_now} · Chênh lệch: {difference}.",
            })

    if anomalies:
        data_table(
            [
                TableColumn("level", "Mức độ", 120),
                TableColumn("type", "Tín hiệu", 230),
                TableColumn("person", "Đối tượng", 210),
                TableColumn("detail", "Chi tiết", 430),
            ],
            [{
                "level": status_badge(
                    row["level"], "danger" if row["level"] == "An ninh" else "warning",
                ),
                "type": row["type"], "person": row["person"], "detail": row["detail"],
            } for row in anomalies],
            compact=True, max_height=480,
        )
    else:
        data_table([], [], empty_message="Không phát hiện trường hợp cần xác minh trong khoảng đã chọn.")


def _space_history(
    density_rows: list[dict], transitions: list[dict], visits: list[dict],
    zones: tuple[Zone, ...],
) -> None:
    zone_labels = {zone.zone_id: zone.label for zone in zones}
    st.write("")
    if density_rows:
        density = pd.DataFrame(density_rows)
        density["captured_at"] = pd.to_datetime(density["captured_at"])
        density["Vùng"] = density["zone_id"].map(zone_labels).fillna(density["zone_id"])
        density["Mốc thời gian"] = density["captured_at"].dt.floor("15min")
        timeline = density.groupby(["Mốc thời gian", "camera_id", "Vùng"], as_index=False)["people_count"].mean()
        timeline["Số người"] = timeline["people_count"].round(1)
        section_header("Mật độ theo thời gian", "Trung bình theo vùng và camera")
        chart = (
            alt.Chart(timeline).mark_line(point=True, strokeWidth=2).encode(
                x=alt.X("Mốc thời gian:T", title=None),
                y=alt.Y("Số người:Q", title="Số người trung bình", scale=alt.Scale(zero=True)),
                color=alt.Color("Vùng:N", legend=alt.Legend(orient="top")),
                strokeDash=alt.StrokeDash("camera_id:N", title="Camera"),
                tooltip=["camera_id:N", "Vùng:N", "Mốc thời gian:T", "Số người:Q"],
            ).properties(height=300)
        )
        st.altair_chart(chart, width="stretch")

        left, right = st.columns([1.35, 1], gap="large")
        with left:
            section_header("Heatmap mật độ", "Mật độ trung bình theo giờ và vùng")
            density["Giờ"] = density["captured_at"].dt.hour
            heatmap = density.groupby(["Giờ", "Vùng"], as_index=False)["people_count"].mean()
            heatmap["Mật độ"] = heatmap["people_count"].round(1)
            heat = alt.Chart(heatmap).mark_rect(cornerRadius=4).encode(
                x=alt.X("Giờ:O", title="Giờ trong ngày"), y=alt.Y("Vùng:N", title=None),
                color=alt.Color("Mật độ:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["Vùng:N", "Giờ:O", "Mật độ:Q"],
            ).properties(height=230)
            st.altair_chart(heat, width="stretch")
        with right:
            section_header("Thời gian lưu trú", "Trung bình các lượt đã rời vùng")
            if visits:
                visit_frame = pd.DataFrame(visits)
                visit_frame["Vùng"] = visit_frame["zone_id"].map(zone_labels).fillna(visit_frame["zone_id"])
                dwell = visit_frame.groupby("Vùng", as_index=False)["dwell_seconds"].mean()
                dwell["Phút"] = (dwell["dwell_seconds"] / 60).round(1)
                bars = alt.Chart(dwell).mark_bar(color="#10b981", cornerRadiusEnd=5).encode(
                    x=alt.X("Phút:Q", title="Phút trung bình"),
                    y=alt.Y("Vùng:N", title=None, sort="-x"), tooltip=["Vùng:N", "Phút:Q"],
                ).properties(height=230)
                st.altair_chart(bars, width="stretch")
            else:
                st.info("Chưa có lượt rời vùng để tính thời gian lưu trú.")
    else:
        st.info("Chưa có mẫu mật độ trong khoảng thời gian đã chọn.")

    section_header("Luồng di chuyển gần đây", "Sự kiện đổi vùng; không lưu video thô")
    if transitions:
        data_table(
            [
                TableColumn("time", "Thời gian", 165), TableColumn("camera", "Camera", 160),
                TableColumn("employee", "Nhân viên", 190), TableColumn("track", "Track", 75, "center"),
                TableColumn("flow", "Luồng di chuyển", 260),
            ],
            [{
                "time": pd.to_datetime(row["event_time"]).strftime("%d/%m/%Y %H:%M:%S"),
                "camera": row["camera_id"], "employee": row["employee_name"],
                "track": f'#{row["track_id"]}',
                "flow": f'{zone_labels.get(row["from_zone"], "Ngoài vùng")} → '
                        f'{zone_labels.get(row["to_zone"], "Ngoài vùng")}',
            } for row in transitions[:80]], compact=True, max_height=390,
        )
    else:
        data_table([], [], empty_message="Chưa có sự kiện di chuyển.")


@st.fragment(run_every=0.8)
def _live_fragment(engines: dict[str, SpatialAnalyticsEngine], zones: tuple[Zone, ...]) -> None:
    columns = st.columns(min(2, max(1, len(engines))), gap="large")
    zone_labels = {zone.zone_id: zone.label for zone in zones}
    for index, (name, engine) in enumerate(engines.items()):
        with columns[index % len(columns)]:
            frame, tracks = engine.latest()
            counts = engine.current_counts()
            state = "Trực tuyến" if engine.camera.connected else "Đang kết nối lại"
            st.markdown(
                f'<div class="spatial-camera-head"><span><i class="bi bi-camera-video-fill"></i>'
                f'{escape(name)}</span><b>{escape(state)}</b></div>', unsafe_allow_html=True,
            )
            if engine.error:
                st.warning(engine.error)
            if frame is not None:
                st.image(frame, channels="BGR", width="stretch")
            else:
                st.info("Đang chờ khung hình đầu tiên…")
            chips = "".join(
                f'<span>{escape(zone_labels.get(zone_id, zone_id))}: <b>{count}</b></span>'
                for zone_id, count in counts.items()
            )
            st.markdown(
                f'<div class="spatial-density-chips">{chips}'
                f'<small>AI: {engine.last_inference_ms or "—"} ms</small></div>', unsafe_allow_html=True,
            )
            if tracks:
                data_table(
                    [TableColumn("track", "Track", 70, "center"),
                     TableColumn("employee", "Nhân viên", 170), TableColumn("zone", "Vùng", 120)],
                    [{"track": f"#{track.track_id}", "employee": track.employee_name,
                      "zone": status_badge(zone_labels.get(engine.track_zone(track.track_id), "Ngoài vùng"),
                                           "success" if track.employee_id else "info")}
                     for track in tracks], compact=True, max_height=230,
                )


def _camera_presence_timeline(
    rows: list[dict], selected_day: date, bucket_minutes: int,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Mốc thời gian", "Số người", "Nguồn"])
    frame = pd.DataFrame(rows)
    frame["captured_at"] = pd.to_datetime(frame["captured_at"])
    comparable = frame["captured_at"].dt.tz_localize(None) if frame["captured_at"].dt.tz is not None else frame["captured_at"]
    frame = frame[comparable.dt.date == selected_day]
    if frame.empty:
        return pd.DataFrame(columns=["Mốc thời gian", "Số người", "Nguồn"])
    frame["Mốc thời gian"] = frame["captured_at"].dt.floor(f"{bucket_minutes}min")
    per_camera = frame.groupby(["Mốc thời gian", "camera_id"], as_index=False)["people_count"].sum()
    result = per_camera.groupby("Mốc thời gian", as_index=False)["people_count"].sum()
    result = result.rename(columns={"people_count": "Số người"})
    result["Nguồn"] = "Camera"
    return result


def _zone_legend(zones: tuple[Zone, ...]) -> None:
    items = "".join(
        f'<span><i style="background:rgb({zone.color[2]},{zone.color[1]},{zone.color[0]})"></i>'
        f'{escape(zone.label)}</span>' for zone in zones
    )
    st.markdown(
        f'<div class="spatial-zone-legend"><b>Vùng camera</b>{items}'
        '<small>Cần hiệu chỉnh polygon theo góc camera thực tế</small></div>', unsafe_allow_html=True,
    )


def _start_engines(
    engines: dict[str, SpatialAnalyticsEngine], detector: PersonDetector,
    repository: SpatialRepository, zones: tuple[Zone, ...],
    face_detector: FaceDetector, recognizer: FaceRecognizer,
) -> None:
    recognizer.reload()
    for name, source in settings.camera_sources:
        engines[name] = SpatialAnalyticsEngine(
            name, CameraManager(source), detector, repository, zones, face_detector, recognizer,
        ).start()
    st.session_state.spatial_engines = engines


def _stop_engines(engines: dict[str, SpatialAnalyticsEngine]) -> None:
    for engine in engines.values():
        engine.stop()
    st.session_state.spatial_engines = {}


def _default_dates(rows: list[dict]) -> tuple[date, date]:
    dates = sorted(date.fromisoformat(str(row["date"])) for row in rows if row.get("date"))
    if not dates:
        return date.today() - timedelta(days=6), date.today()
    end = min(date.today(), dates[-1])
    return max(dates[0], end - timedelta(days=6)), end


def _date_range(value, fallback_start: date, fallback_end: date) -> tuple[date, date]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return min(value), max(value)
    if isinstance(value, date):
        return value, value
    return fallback_start, fallback_end


def _filter_timestamp_rows(
    rows: list[dict], field: str, start_day: date, end_day: date,
) -> list[dict]:
    start = pd.Timestamp(start_day)
    end = pd.Timestamp(end_day) + pd.Timedelta(days=1)
    result = []
    for row in rows:
        timestamp = pd.to_datetime(row.get(field), errors="coerce")
        if pd.isna(timestamp):
            continue
        comparable = timestamp.tz_localize(None) if timestamp.tzinfo else timestamp
        if start <= comparable < end:
            result.append(row)
    return result
