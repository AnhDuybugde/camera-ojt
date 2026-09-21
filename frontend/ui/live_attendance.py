from __future__ import annotations

import json
import streamlit as st
from html import escape
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from attendance.attendance_service import AttendanceService
from auth.permissions import require_permission
from camera.camera_manager import CameraManager
from camera.live_engine import LiveAttendanceEngine
from config import settings
from face.detector import FaceDetector
from face.recognizer import FaceRecognizer
from ui.components import (
    TableColumn, data_table, employee_cell, page_header, section_header,
    status_badge, translate_message,
)


def render(detector: FaceDetector, recognizer: FaceRecognizer,
           attendance: AttendanceService, role: str) -> None:
    require_permission(role, "camera.view")
    if settings.tracking_backend_url:
        _render_tracking_backend(settings.tracking_backend_url)
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


def _render_tracking_backend(base_url: str) -> None:
    """Render the shared model service without opening the cameras again."""
    page_header(
        "Camera điểm danh trực tiếp",
        "Luồng dùng chung từ YOLO · ByteTrack · Global Re-ID; giao diện không chiếm RTSP lần hai",
        status="Camera OJT backend",
    )
    safe_base = escape(base_url.rstrip("/"), quote=True)
    with st.container(border=True):
        st.caption(f"Nguồn model: {base_url}/status.json")
        left, right = st.columns(2)
        with left:
            st.markdown(
                f'<div class="stitch-camera-title">Camera A</div>'
                f'<img src="{safe_base}/cam_a.mjpg" '
                'style="width:100%;border-radius:12px;background:#0b1220" '
                'alt="Camera A" />',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f'<div class="stitch-camera-title">Camera B</div>'
                f'<img src="{safe_base}/cam_b.mjpg" '
                'style="width:100%;border-radius:12px;background:#0b1220" '
                'alt="Camera B" />',
                unsafe_allow_html=True,
            )
    _tracking_backend_fragment(base_url)


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


@st.fragment(run_every=0.5)
def _tracking_backend_fragment(base_url: str) -> None:
    try:
        payload = _fetch_tracking_status(base_url)
    except (HTTPError, URLError, OSError, TypeError, ValueError) as error:
        st.warning(f"Đang chờ Camera OJT backend: {error}")
        return

    people = payload.get("people", [])
    if not isinstance(people, list):
        people = []
    count_a = int(payload.get("count_a") or 0)
    count_b = int(payload.get("count_b") or 0)
    metric_a, metric_b, metric_total = st.columns(3)
    metric_a.metric("Camera A", count_a)
    metric_b.metric("Camera B", count_b)
    metric_total.metric("Tổng hiện diện", count_a + count_b)

    rows = []
    for person in people:
        if not isinstance(person, dict):
            continue
        person_id = str(person.get("person_id") or "UNKNOWN")
        name = str(person.get("name") or "Chưa xác định")
        score = person.get("face_score")
        rows.append({
            "employee": employee_cell(name, person_id, f"Global ID {person.get('gid', '—')}"),
            "camera": escape(str(person.get("camera") or "—")),
            "state": status_badge(
                "Trong phòng" if person.get("in_room") else "Ngoài phòng",
                "success" if person.get("in_room") else "neutral",
            ),
            "score": "—" if score is None else f"{float(score):.1%}",
        })
    if rows:
        data_table(
            [
                TableColumn("employee", "Nhân viên", 240),
                TableColumn("camera", "Camera", 80, "center"),
                TableColumn("state", "Trạng thái", 130),
                TableColumn("score", "Face score", 100, "center"),
            ],
            rows,
            compact=True,
        )
    else:
        st.info("Backend đang hoạt động nhưng chưa có người được nhận diện.")


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
