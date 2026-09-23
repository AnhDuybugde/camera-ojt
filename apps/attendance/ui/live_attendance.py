from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
from html import escape

from attendance.attendance_service import AttendanceService
from auth.permissions import require_permission
from camera.camera_manager import CameraManager
from camera.live_engine import LiveAttendanceEngine
from config import settings
from face.detector import FaceDetector
from face.recognizer import FaceRecognizer
from integration.camera_ojt import CameraOjtClient
from ui.components import (
    TableColumn, data_table, employee_cell, page_header, section_header,
    status_badge, translate_message,
)


def render(detector: FaceDetector, recognizer: FaceRecognizer,
           attendance: AttendanceService, role: str) -> None:
    require_permission(role, "camera.view")
    if settings.camera_ojt_url:
        _render_camera_ojt()
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


def _render_camera_ojt() -> None:
    client = CameraOjtClient(settings.camera_ojt_url, settings.camera_ojt_supervisor_url)
    try:
        status = client.status()
        cameras = status.get("cameras") if isinstance(status.get("cameras"), list) else []
        connected = sum(bool(item.get("live")) for item in cameras if isinstance(item, dict))
        online = True
    except RuntimeError as error:
        status, connected, online = {}, 0, False
        connection_error = str(error)
    page_header(
        "Camera điểm danh trực tiếp",
        "Luồng và kết quả nhận diện từ backend camera-ojt · không xử lý RTSP trùng lặp",
        status=f"{connected}/2 camera trực tuyến" if online else "Backend ngoại tuyến",
    )
    if not online:
        st.error(connection_error)
        st.info("Khởi động backend camera-ojt; trang này sẽ tự nhận stream và kết quả nhận diện.")
        return
    channel = st.segmented_control(
        "Camera hiển thị", ["Camera A", "Camera B"], default="Camera A",
        key="camera_ojt_channel", width="stretch",
    ) or "Camera A"
    key = "B" if channel.endswith("B") else "A"
    stream_url = client.stream_url(key)
    components.html(
        f'<div style="background:#101318;border-radius:12px;overflow:hidden;aspect-ratio:16/9">'
        f'<img src="{stream_url}" alt="{channel}" style="width:100%;height:100%;object-fit:contain;display:block" />'
        f'</div>', height=520, scrolling=False,
    )
    people = [item for item in status.get("people", []) if isinstance(item, dict)]
    data_table(
        [
            TableColumn("employee", "Nhân viên", 230),
            TableColumn("camera", "Camera", 90, "center"),
            TableColumn("state", "Trạng thái", 150),
            TableColumn("confidence", "Độ tin cậy", 100, "center"),
        ],
        [{
            "employee": employee_cell(item.get("name") or "Chưa xác định", item.get("person_id") or f"G{item.get('gid', '?')}", "camera-ojt"),
            "camera": item.get("camera") or "—",
            "state": status_badge(str(item.get("label") or "Unknown"), "success" if item.get("in_room") else "neutral"),
            "confidence": f"{float(item.get('face_score')):.1%}" if item.get("face_score") is not None else "—",
        } for item in people],
        empty_message="Chưa phát hiện người nào trong khung hình.", compact=True,
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
