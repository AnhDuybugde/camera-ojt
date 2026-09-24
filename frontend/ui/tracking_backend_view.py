from __future__ import annotations

import json
import streamlit as st
from datetime import date, datetime
from html import escape
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from attendance.attendance_service import AttendanceService
from auth.permissions import ADMIN, require_permission
from camera.camera_manager import CameraManager
from camera.live_engine import LiveAttendanceEngine
from config import settings
from face.detector import FaceDetector
from face.recognizer import FaceRecognizer
from utils.be_xinh_control import BeXinhControlError, BeXinhController
from utils.system_control import (
    SystemControlError,
    SystemController,
    VoiceVolumeSettings,
)
from ui.components import (
    TableColumn, data_table, employee_cell, page_header, section_header,
    status_badge, translate_message,
)


def render(detector: FaceDetector, recognizer: FaceRecognizer,
           attendance: AttendanceService, role: str) -> None:
    require_permission(role, "camera.view")
    if settings.tracking_backend_url:
        _render_tracking_backend(settings.tracking_backend_url, role)
        return
    if "live_engines" not in st.session_state:
        st.session_state.live_engines = {}
    engines: dict[str, LiveAttendanceEngine] = st.session_state.live_engines
    connected = sum(bool(engine.camera.connected) for engine in engines.values())
    page_header(
        "Camera điểm danh trực tiếp",
        "Nhận diện đồng thời các luồng RTSP đã cấu hình và cập nhật SQLite tức thời",
        status=f"{connected}/{len(settings.camera_sources)} camera trực tuyến",
    )

    with st.container(border=True):
        left, right, info = st.columns([1, 1, 3])
        if left.button(
            "Khởi động camera",
            type="primary",
            disabled=bool(engines),
            width="stretch",
            icon=":material/play_arrow:",
        ):
            _start_engines(engines, detector, recognizer, attendance)
            st.rerun()
        if right.button(
            "Dừng tất cả",
            disabled=not engines,
            width="stretch",
            icon=":material/stop:",
        ):
            _stop_engines(engines)
            st.rerun()
        info.caption(
            f"{len(settings.camera_sources)} luồng camera · "
            f"Giao thức RTSP {settings.rtsp_transport.upper()} · "
            f"Nhận diện mỗi {settings.recognition_interval:.2f} giây"
        )

    if not engines:
        st.markdown(
            '<div class="stitch-empty">'
            '  <div class="stitch-empty-icon">'
            '    <i class="bi bi-camera-video-off-fill" aria-hidden="true"></i>'
            '  </div>'
            '  <h3>Camera đang dừng</h3>'
            '  <p>Nhấn "Khởi động camera" để bắt đầu nhận diện và điểm danh.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    section_header("Giám sát trực tiếp", "Khung hình mới nhất · nhận diện chạy nền")
    _live_fragment(engines)


def render_tracking_backend(base_url: str, role: str) -> None:
    """Render the shared model service without opening the cameras again."""
    page_header(
        "Camera điểm danh trực tiếp",
        "Luồng dùng chung từ YOLO · ByteTrack · Global Re-ID; giao diện không chiếm RTSP lần hai",
        status="Camera OJT backend",
    )
    _system_control_fragment(role)
    browser_base = settings.public_tracking_backend_url or base_url
    with st.container(border=True):
        st.caption(f"Nguồn model: {base_url}/status.json")
        view_mode = st.radio(
            "Kích thước camera",
            ("Xem cả hai", "Phóng to Camera A", "Phóng to Camera B"),
            horizontal=True,
            key="tracking_camera_view_mode",
            help="Phóng một camera ra toàn chiều ngang để quan sát và gắn nhãn dễ hơn.",
        )

        if view_mode == "Phóng to Camera A":
            _render_tracking_camera(browser_base, "A", "cam_a.mjpg", expanded=True)
        elif view_mode == "Phóng to Camera B":
            _render_tracking_camera(browser_base, "B", "cam_b.mjpg", expanded=True)
        else:
            left, right = st.columns(2)
            with left:
                _render_tracking_camera(browser_base, "A", "cam_a.mjpg")
            with right:
                _render_tracking_camera(browser_base, "B", "cam_b.mjpg")
    _tracking_backend_fragment(base_url)


# Backward compatibility for older imports and hot-reloaded Streamlit sessions.
_render_tracking_backend = render_tracking_backend


def _render_tracking_camera(
    browser_base: str,
    camera_name: str,
    endpoint: str,
    *,
    expanded: bool = False,
) -> None:
    """Render one shared MJPEG stream without opening another RTSP connection."""
    safe_base = escape(browser_base.rstrip("/"), quote=True)
    safe_name = escape(camera_name, quote=True)
    safe_endpoint = escape(endpoint.lstrip("/"), quote=True)
    stream_url = f"{safe_base}/{safe_endpoint}"
    mode_label = " · chế độ phóng to" if expanded else ""
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'gap:12px;margin:2px 0 6px">'
        f'<div class="stitch-camera-title">Camera {safe_name}{mode_label}</div>'
        f'<a href="{stream_url}" target="_blank" rel="noopener noreferrer" '
        'style="font-size:.88rem;text-decoration:none">Mở trong tab riêng ↗</a>'
        '</div>'
        f'<img src="{stream_url}" '
        'style="display:block;width:100%;height:auto;border-radius:12px;'
        'background:#0b1220;object-fit:contain" '
        f'alt="Camera {safe_name}" />',
        unsafe_allow_html=True,
    )


@st.fragment(run_every=2.0)
def _system_control_fragment(role: str) -> None:
    """Persistent control center; only admins may mutate service state."""
    controller = SystemController()
    camera = controller.backend_status()
    voice = controller.chat_status()
    can_control = str(role).upper() == ADMIN

    with st.container(border=True):
        st.markdown("### Điều khiển hệ thống")
        st.caption(
            "Dashboard hoạt động độc lập. Tắt Camera AI hoặc Bé Xinh sẽ không làm mất trang web."
        )
        camera_col, voice_col = st.columns(2)

        with camera_col:
            if camera.healthy:
                st.success(f"Camera AI · Đang bật · PID {camera.pid}", icon=":material/videocam:")
            elif camera.running:
                st.warning(f"Camera AI · {camera.detail} · PID {camera.pid}", icon=":material/sync:")
            else:
                st.info("Camera AI · Đã tắt", icon=":material/videocam_off:")
            start, stop, restart = st.columns(3)
            camera_action = None
            if start.button("Bật", key="system_camera_start", disabled=not can_control or camera.running, width="stretch"):
                camera_action = controller.start_backend
            if stop.button("Tắt", key="system_camera_stop", disabled=not can_control or not camera.running, width="stretch"):
                camera_action = controller.stop_backend
            if restart.button("Khởi động lại", key="system_camera_restart", disabled=not can_control, width="stretch"):
                camera_action = controller.restart_backend
            if camera_action:
                _run_system_action(camera_action, "Camera AI")

        with voice_col:
            if voice.running:
                st.success(f"Bé Xinh · Đang bật · PID {voice.pid}", icon=":material/record_voice_over:")
            else:
                st.info("Bé Xinh · Đã tắt", icon=":material/voice_over_off:")
            start, stop, restart = st.columns(3)
            voice_action = None
            if start.button("Bật", key="system_voice_start", disabled=not can_control or voice.running, width="stretch"):
                voice_action = controller.start_chat
            if stop.button("Tắt", key="system_voice_stop", disabled=not can_control or not voice.running, width="stretch"):
                voice_action = controller.stop_chat
            if restart.button("Khởi động lại", key="system_voice_restart", disabled=not can_control, width="stretch"):
                voice_action = controller.restart_chat
            if voice_action:
                _run_system_action(voice_action, "Bé Xinh")

        volume = controller.voice_volume()
        st.caption(f"Âm lượng Bé Xinh hiện tại: {volume.percent}%")
        if can_control:
            with st.expander("Chỉnh âm lượng Bé Xinh", expanded=False):
                with st.form("be_xinh_volume_form"):
                    selected_percent = st.slider(
                        "Âm lượng Bé Xinh", 10, 100,
                        value=volume.percent, step=5,
                    )
                    if st.form_submit_button(
                        "Lưu âm lượng", type="primary", width="stretch"
                    ):
                        controller.save_voice_volume(VoiceVolumeSettings(
                            percent=selected_percent,
                        ))
                        st.toast("Đã lưu âm lượng Bé Xinh; áp dụng từ câu nói kế tiếp.")
                        st.rerun()

        if not can_control:
            st.caption("Bạn đang ở chế độ xem. Chỉ tài khoản Quản trị viên được bật/tắt dịch vụ.")


def _run_system_action(action, label: str) -> None:
    try:
        action()
    except SystemControlError as error:
        st.error(f"{label}: {error}")
    else:
        st.toast(f"Đã cập nhật {label}")
        st.rerun()


@st.fragment(run_every=2.0)
def _be_xinh_control_fragment() -> None:
    """Show a verified on/off control for the independent audio bridge."""
    controller = BeXinhController()
    current = controller.status()

    with st.container(border=True):
        title, state = st.columns([4, 1])
        title.markdown("### Bé Xinh · Trợ lý giọng nói")
        if current.running:
            state.success("Đang bật", icon=":material/volume_up:")
            title.caption("Bé Xinh có thể chào khi thấy cử chỉ và nhắc nghỉ ngơi theo ngữ cảnh.")
        else:
            state.info("Đã tắt", icon=":material/volume_off:")
            title.caption("Camera và nhận diện vẫn hoạt động; chỉ phần giọng nói đang tắt.")

        turn_on, turn_off, note = st.columns([1, 1, 3])
        if turn_on.button(
            "Bật Bé Xinh",
            type="primary",
            disabled=current.running,
            icon=":material/play_arrow:",
            width="stretch",
            key="be_xinh_on",
        ):
            try:
                controller.start()
            except BeXinhControlError as error:
                st.error(str(error))
            else:
                st.toast("Đã bật Bé Xinh", icon="🔊")
                st.rerun()
        if turn_off.button(
            "Tắt Bé Xinh",
            disabled=not current.running,
            icon=":material/stop:",
            width="stretch",
            key="be_xinh_off",
        ):
            try:
                controller.stop()
            except BeXinhControlError as error:
                st.error(str(error))
            else:
                st.toast("Đã tắt Bé Xinh", icon="🔇")
                st.rerun()
        note.caption("Nút này không dừng camera, model nhận diện hoặc giao diện điểm danh.")


def _fetch_tracking_status(base_url: str) -> dict:
    request = Request(
        f"{base_url.rstrip('/')}/status.json",
        headers={"Accept": "application/json", "User-Agent": "ai-mind-ui/1"},
    )
    with urlopen(request, timeout=2.0) as response:  # noqa: S310
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise TypeError("backend status must be a JSON object")
    return payload


def _active_tracking_people(payload: dict) -> list[dict]:
    """Exclude retained/lost IDs and people no longer visible in a camera."""
    raw_people = payload.get("people", [])
    if not isinstance(raw_people, list):
        return []
    return [
        person
        for person in raw_people
        if isinstance(person, dict)
        and (
            bool(person.get("visible"))
            if "visible" in person
            else str(person.get("tracking_state") or "ACTIVE").upper() == "ACTIVE"
        )
    ]


_WORK_STATE_LABELS = {
    "Working": "Đang làm việc",
    "Near seat": "Gần vị trí làm việc",
    "Away": "Tạm rời vị trí",
    "Out of office": "Đã ra ngoài",
    "Returning": "Đang quay lại",
    "Unknown": "Chưa đủ dữ liệu",
}

_WORK_STATE_TONES = {
    "Working": "success",
    "Near seat": "info",
    "Away": "warning",
    "Out of office": "neutral",
    "Returning": "primary",
    "Unknown": "neutral",
}


def _attendance_overview(payload: dict) -> list[dict]:
    """Join today's attendance with the complete face-gallery roster."""
    today = date.today().isoformat()
    attendance_by_id: dict[str, dict] = {}
    for key in ("attendance_today", "pending_attendance"):
        items = payload.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or item.get("attended") is False:
                continue
            item_day = str(item.get("date") or today)
            person_id = str(item.get("person_id") or "").strip()
            if not person_id or item_day != today:
                continue
            attendance_by_id.setdefault(person_id, item)

    roster_items = payload.get("employee_roster", [])
    roster: dict[str, dict] = {}
    if isinstance(roster_items, list):
        for item in roster_items:
            if not isinstance(item, dict):
                continue
            person_id = str(item.get("person_id") or "").strip()
            if person_id:
                roster[person_id] = item
    # Backward-compatible fallback while an older backend is restarting.
    if not roster:
        for item in [*attendance_by_id.values(), *payload.get("people", [])]:
            if not isinstance(item, dict):
                continue
            person_id = str(item.get("person_id") or "").strip()
            if not person_id:
                continue
            roster.setdefault(person_id, {
                "person_id": person_id,
                "person_name": item.get("person_name") or item.get("name") or person_id,
                "gallery_samples": 0,
            })

    rows = []
    for person_id, person in roster.items():
        attendance = attendance_by_id.get(person_id)
        rows.append({
            "person_id": person_id,
            "person_name": str(person.get("person_name") or person_id),
            "gallery_samples": int(person.get("gallery_samples") or 0),
            "checked_in": attendance is not None,
            "check_in_at": attendance.get("check_in_at") if attendance else None,
            "face_score": attendance.get("face_score") if attendance else None,
        })
    return sorted(
        rows,
        key=lambda item: (not item["checked_in"], item["person_name"].casefold()),
    )


def _checkin_time(value: object) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%H:%M:%S")
    except (TypeError, ValueError):
        return str(value)


@st.fragment(run_every=0.5)
def _tracking_backend_fragment(base_url: str) -> None:
    try:
        payload = _fetch_tracking_status(base_url)
    except (HTTPError, URLError, OSError, TypeError, ValueError) as error:
        st.warning(f"Đang chờ Camera OJT backend: {error}")
        return

    people = _active_tracking_people(payload)
    attendance_rows = _attendance_overview(payload)
    checked_ids = {
        row["person_id"] for row in attendance_rows if row["checked_in"]
    }
    count_a = int(payload.get("count_a") or 0)
    count_b = int(payload.get("count_b") or 0)
    raw_count_total = payload.get("count")
    count_total = (
        count_a + count_b
        if raw_count_total is None
        else int(raw_count_total)
    )
    metric_a, metric_b, metric_total, metric_checked = st.columns(4)
    metric_a.metric("Camera A", count_a)
    metric_b.metric("Camera B", count_b)
    metric_total.metric("Tổng hiện diện", count_total)
    metric_checked.metric("Đã check-in", len(checked_ids))

    present_tab, checkin_tab, states_tab = st.tabs([
        "Đang hiện diện", "Check-in hôm nay", "Ý nghĩa trạng thái"
    ])
    with present_tab:
        rows = []
        for person in people:
            person_id = str(person.get("person_id") or "UNKNOWN")
            name = str(person.get("name") or "Chưa xác định")
            score = person.get("face_score")
            work_state = str(person.get("label") or "Unknown")
            is_known = person_id != "UNKNOWN"
            rows.append({
                "employee": employee_cell(
                    name, person_id, f"Global ID {person.get('gid', '—')}"
                ),
                "camera": str(person.get("camera") or "—"),
                "activity": status_badge(
                    _WORK_STATE_LABELS.get(work_state, work_state),
                    _WORK_STATE_TONES.get(work_state, "neutral"),
                ),
                "attendance": status_badge(
                    "Đã check-in" if person_id in checked_ids else (
                        "Chưa check-in" if is_known else "Chưa nhận diện"
                    ),
                    "success" if person_id in checked_ids else (
                        "warning" if is_known else "neutral"
                    ),
                ),
                "score": "—" if score is None else f"{float(score):.1%}",
            })
        data_table(
            [
                TableColumn("employee", "Nhân viên", 240),
                TableColumn("camera", "Camera", 75, "center"),
                TableColumn("activity", "Hoạt động", 155),
                TableColumn("attendance", "Điểm danh", 140),
                TableColumn("score", "Face score", 95, "center"),
            ],
            rows,
            compact=True,
            empty_message="Camera đang hoạt động nhưng hiện chưa thấy người.",
        )

    with checkin_tab:
        total = len(attendance_rows)
        weak_gallery = sum(
            row["gallery_samples"] < 3 for row in attendance_rows
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng nhân viên", total)
        c2.metric("Chưa check-in", max(0, total - len(checked_ids)))
        c3.metric("Cần bổ sung ảnh mẫu", weak_gallery)
        data_table(
            [
                TableColumn("employee", "Nhân viên", 250),
                TableColumn("attendance", "Trạng thái hôm nay", 160),
                TableColumn("time", "Giờ check-in", 110, "center"),
                TableColumn("score", "Face score", 95, "center"),
                TableColumn("gallery", "Dữ liệu khuôn mặt", 155),
            ],
            [{
                "employee": employee_cell(
                    row["person_name"], row["person_id"], "Hồ sơ nhận diện"
                ),
                "attendance": status_badge(
                    "Đã check-in" if row["checked_in"] else "Chưa check-in",
                    "success" if row["checked_in"] else "warning",
                ),
                "time": _checkin_time(row["check_in_at"]),
                "score": (
                    "—" if row["face_score"] is None
                    else f"{float(row['face_score']):.1%}"
                ),
                "gallery": status_badge(
                    f"{row['gallery_samples']} mẫu · " + (
                        "Tốt" if row["gallery_samples"] >= 3 else "Cần bổ sung"
                    ),
                    "success" if row["gallery_samples"] >= 3 else "warning",
                ),
            } for row in attendance_rows],
            compact=True,
            empty_message="Chưa có hồ sơ nhân viên trong thư viện khuôn mặt.",
        )

    with states_tab:
        st.markdown(
            "**Có 6 trạng thái hoạt động:** `Working`, `Near seat`, `Away`, "
            "`Out of office`, `Returning`, `Unknown`.\n\n"
            "`(Unknown)` nằm cạnh Global ID là **chưa nhận ra danh tính**; "
            "còn `Unknown` ở cuối dòng là **chưa đủ dữ liệu hoạt động**. "
            "Hai trạng thái này độc lập với nhau."
        )
        state_rows = [
            ("Working", "Đang làm việc / đang ở vị trí ổn định"),
            ("Near seat", "Đang ở gần vị trí làm việc"),
            ("Away", "Tạm rời vị trí nhưng vẫn được xem là trong phòng"),
            ("Out of office", "Đã rời khỏi văn phòng"),
            ("Returning", "Vừa quay lại, hệ thống đang xác nhận ổn định"),
            ("Unknown", "Chưa đủ quan sát để kết luận"),
        ]
        data_table(
            [
                TableColumn("state", "Nhãn kỹ thuật", 150),
                TableColumn("meaning", "Ý nghĩa", 390),
            ],
            [{"state": status_badge(
                label, _WORK_STATE_TONES.get(label, "neutral")
            ), "meaning": meaning} for label, meaning in state_rows],
            compact=True,
        )


def _start_engines(
    engines: dict[str, LiveAttendanceEngine],
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    attendance: AttendanceService,
) -> None:
    recognizer.reload()
    for name, source in settings.camera_sources:
        engines[name] = LiveAttendanceEngine(
            CameraManager(source), detector, recognizer, attendance
        ).start()
    st.session_state.live_engines = engines


def _stop_engines(engines: dict[str, LiveAttendanceEngine]) -> None:
    for engine in engines.values():
        engine.stop()
    st.session_state.live_engines = {}


@st.fragment(run_every=0.15)
def _live_fragment(engines: dict[str, LiveAttendanceEngine]) -> None:
    columns = st.columns(min(2, len(engines)))
    for index, (name, engine) in enumerate(engines.items()):
        with columns[index % len(columns)]:
            connected = engine.camera.connected
            state = "TRỰC TUYẾN" if connected else "MẤT KẾT NỐI · ĐANG THỬ LẠI"
            conn_icon = "bi-link-45deg" if connected else "bi-link-slash"

            # Camera header
            st.markdown(
                f'<div class="stitch-card" style="padding:var(--sp-sm) var(--sp-md);margin-bottom:0;">'
                f'  <div class="stitch-camera-header" style="margin-bottom:0;padding-bottom:6px;">'
                f'    <div>'
                f'      <div class="stitch-camera-title" style="font-size:14px;">'
                f'        <span class="stitch-camera-pulse"></span>'
                f'        IMOU · {escape(name)}'
                f'      </div>'
                f'    </div>'
                f'    <span class="stitch-conn-badge">'
                f'      <i class="bi {conn_icon}"></i> {state}'
                f'    </span>'
                f'  </div>',
                unsafe_allow_html=True,
            )

            frame, detections = engine.latest()
            if engine.error:
                st.warning(translate_message(engine.error))
            if frame is not None:
                # Streamlit accepts OpenCV's native BGR frames directly. Avoiding
                # cvtColor here also prevents sporadic OpenCV C++ failures while
                # the recognition worker is using the same runtime concurrently.
                st.image(frame, channels="BGR", width="stretch")
            else:
                st.info("Đang chờ khung hình đầu tiên…")

            if detections:
                data_table(
                    [
                        TableColumn("employee", "Nhân viên", 210),
                        TableColumn("similarity", "Tương đồng", 90, "center"),
                        TableColumn("result", "Kết quả", 160),
                    ],
                    [{
                        "employee": employee_cell(
                            item.name, item.employee_id or "UNKNOWN", "Nhận diện trực tiếp"
                        ),
                        "similarity": f"{item.similarity:.1%}",
                        "result": status_badge(
                            _attendance_label(item.attendance),
                            _attendance_tone(item.attendance),
                        ),
                    } for item in detections],
                    compact=True,
                )

            age = engine.camera.frame_age_ms
            st.markdown(
                f'<div class="stitch-camera-overlay">'
                f'  <span>{len(detections)} khuôn mặt</span>'
                f'  <span>Khung hình: {age if age is not None else "—"} ms'
                f'  · AI: {engine.last_inference_ms or "—"} ms</span>'
                f'</div>'
                f'</div>',   # close stitch-card
                unsafe_allow_html=True,
            )


def _attendance_label(value: str) -> str:
    return {
        "CHECK-IN SUCCESS":                    "Check-in thành công",
        "CHECK-OUT SUCCESS":                   "Check-out thành công",
        "Checked in; checkout is not due yet.":"Đã check-in · chưa đến giờ check-out",
        "Attendance already complete.":        "Đã hoàn tất điểm danh",
        "Already recorded recently.":          "Vừa ghi nhận gần đây",
        "Hôm nay WFH - Không chấm công":       "Hôm nay WFH · Không chấm công",
        "Hôm nay OFF - Không chấm công":       "Hôm nay OFF · Không chấm công",
        "Chưa đăng ký lịch - Không chấm công": "Chưa đăng ký lịch · Không chấm công",
        "Ca này WFH - Không chấm công":         "Ca này WFH · Không chấm công",
        "Ca này OFF - Không chấm công":         "Ca này OFF · Không chấm công",
        "Chưa đăng ký lịch ca này - Không chấm công": "Chưa đăng ký lịch ca này · Không chấm công",
        "Không áp dụng lịch Thứ Bảy/Chủ Nhật - Không chấm công": "Ngoài lịch T2–T6 · Không chấm công",
        "Đang trong giờ nghỉ trưa - Không chấm công": "Đang nghỉ trưa · Không chấm công",
        "UNKNOWN":                             "Không xác định",
        "LIVENESS FAILED":                     "Không đạt kiểm tra người thật",
    }.get(value, value)


def _attendance_tone(value: str) -> str:
    if value in {"CHECK-IN SUCCESS", "CHECK-OUT SUCCESS", "Attendance already complete."}:
        return "success"
    if value in {"UNKNOWN", "LIVENESS FAILED"}:
        return "danger"
    if "WFH" in value:
        return "info"
    if "OFF" in value or "Chưa đăng ký" in value:
        return "neutral"
    return "warning"
