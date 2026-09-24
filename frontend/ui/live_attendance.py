from __future__ import annotations

from datetime import datetime
import streamlit as st
from html import escape
from pathlib import Path

from attendance.attendance_service import AttendanceService
from auth.permissions import require_permission
from camera.camera_manager import CameraManager
from camera.live_engine import DetectionView, LiveAttendanceEngine
from config import settings
from face.detector import FaceDetector
from face.recognizer import FaceRecognizer
from spatial.activity_config import activity_settings
from spatial.activity_engine import ActivityRuntime, ActivityView
from spatial.activity_rules import ActivityType
from spatial.pose_service import PoseService
from spatial.repository import SpatialRepository
from spatial.zones import zones_from_environment
from ui.components import (
    TableColumn, data_table, employee_cell, page_header, section_header,
    status_badge, translate_message,
)
from ui import tracking_backend_view


ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def _activity_repository() -> SpatialRepository:
    return SpatialRepository(ROOT / "data" / "spatial_analytics.db", database_url=settings.database_url)


@st.cache_resource
def _pose_service() -> PoseService:
    return PoseService(activity_settings)


def render(detector: FaceDetector, recognizer: FaceRecognizer,
           attendance: AttendanceService, role: str) -> None:
    require_permission(role, "camera.view")
    if settings.tracking_backend_url:
        tracking_backend_view.render_tracking_backend(
            settings.tracking_backend_url, role
        )
        return
    if "live_engines" not in st.session_state:
        st.session_state.live_engines = {}
    engines: dict[str, LiveAttendanceEngine] = st.session_state.live_engines
    connected = sum(bool(engine.camera.connected) for engine in engines.values())
    page_header(
        "Camera điểm danh trực tiếp",
        "Nhận diện đồng thời các luồng RTSP đã cấu hình và cập nhật điểm danh tức thời",
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

    # This is an explicit departure event. A person disappearing from a face
    # frame is not sufficient evidence to mark them as having left the office.
    today = datetime.now().date().isoformat()
    eligible = [row for row in attendance.db.list_attendance(today, today)
                if row["check_in"] and not row["check_out"]
                and row.get("presence_status") == "PRESENT"]
    with st.expander("Ghi nhận check-out tạm thời"):
        st.caption("Chỉ ghi khi xác nhận nhân viên đã rời văn phòng; camera nhận diện lại sẽ ghi nhận quay lại.")
        if eligible:
            labels = {f'{row["employee_id"]} · {row["employee_name"]}': row["employee_id"]
                      for row in eligible}
            selected = st.selectbox("Nhân viên rời văn phòng", list(labels), key="temp_checkout_employee")
            if st.button("Ghi nhận ra ngoài", key="temp_checkout_confirm"):
                if attendance.daily.mark_temporary_checkout(labels[selected], datetime.now()):
                    st.success("Đã ghi nhận check-out tạm thời.")
                    st.rerun()
                else:
                    st.warning("Không thể ghi nhận lúc này; kiểm tra lịch ON và giờ làm việc.")
        else:
            st.info("Chưa có nhân viên đã check-in và đang hiện diện để ghi nhận ra ngoài.")

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


def _start_engines(
    engines: dict[str, LiveAttendanceEngine],
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    attendance: AttendanceService,
) -> None:
    recognizer.reload()
    repository = _activity_repository()
    pose = _pose_service()
    zones = zones_from_environment()
    for name, source in settings.camera_sources:
        camera = CameraManager(source)
        face_source = _recognition_source(source)
        recognition_camera = camera if face_source == source else CameraManager(face_source)
        activity = ActivityRuntime(name, camera, pose, repository, zones, activity_settings)
        engines[name] = LiveAttendanceEngine(
            camera, detector, recognizer, attendance, activity,
            recognition_camera=recognition_camera,
        ).start()
    st.session_state.live_engines = engines


def _recognition_source(source: int | str) -> int | str:
    """Keep the light sub-stream for UI, use the matching main stream for face crops."""
    if not settings.face_use_main_stream or not isinstance(source, str):
        return source
    return source.replace("subtype=1", "subtype=0")


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
            activities = engine.activity_states()
            if engine.error:
                st.warning(translate_message(engine.error))
            if engine.activity is not None and engine.activity.error:
                st.caption(f"⚠️ {engine.activity.error} · Chấm công vẫn hoạt động bình thường.")
            if frame is not None:
                # Streamlit accepts OpenCV's native BGR frames directly. Avoiding
                # cvtColor here also prevents sporadic OpenCV C++ failures while
                # the recognition worker is using the same runtime concurrently.
                st.image(frame, channels="BGR", width="stretch")
            else:
                st.info("Đang chờ khung hình đầu tiên…")

            if detections or activities:
                data_table(
                    [
                        TableColumn("employee", "Nhân viên", 210),
                        TableColumn("result", "Kết quả", 160),
                        TableColumn("activity", "Hoạt động", 155),
                        TableColumn("duration", "Thời lượng", 85, "center"),
                    ],
                    _combined_camera_rows(detections, activities),
                    compact=True,
                )

            if engine.activity is not None:
                st.caption(
                    f"Camera {engine.camera.capture_fps:.1f} FPS · "
                    f"Face {engine.last_inference_ms or '—'} ms · "
                    f"Activity {engine.activity.activity_fps:.1f} FPS / "
                    f"Pose {engine.activity.last_pose_ms or '—'} ms"
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


def _combined_camera_rows(
    detections: list[DetectionView], activities: list[ActivityView],
) -> list[dict[str, str]]:
    """Merge face and activity results into one table without duplicating tracks."""
    rows: list[dict[str, str]] = []
    used_tracks: set[int] = set()
    for detection in detections:
        candidates = [item for item in activities if item.track_id not in used_tracks]
        if detection.employee_id:
            matched = next(
                (item for item in candidates if item.employee_id == detection.employee_id), None,
            )
        else:
            center_x = (detection.box[0] + detection.box[2]) / 2
            center_y = (detection.box[1] + detection.box[3]) / 2
            containing = [item for item in candidates if (
                item.box[0] <= center_x <= item.box[2]
                and item.box[1] <= center_y <= item.box[3]
            )]
            matched = min(
                containing,
                key=lambda item: abs((item.box[0] + item.box[2]) / 2 - center_x),
                default=None,
            )
        if matched is not None:
            used_tracks.add(matched.track_id)
        rows.append({
            "employee": employee_cell(
                detection.name, detection.employee_id or "UNKNOWN", "Nhận diện trực tiếp",
            ),
            "result": status_badge(
                _attendance_label(detection.attendance),
                _attendance_tone(detection.attendance),
            ),
            "activity": status_badge(
                matched.label if matched else "Không xác định",
                _activity_tone(matched.activity) if matched else "neutral",
            ),
            "duration": _duration(matched.duration_seconds) if matched else "—",
        })

    for item in activities:
        if item.track_id in used_tracks:
            continue
        rows.append({
            "employee": employee_cell(
                item.employee_name, item.employee_id or f"Track #{item.track_id}",
                item.zone or "Ngoài vùng",
            ),
            "result": status_badge("Chưa nhận diện khuôn mặt", "neutral"),
            "activity": status_badge(item.label, _activity_tone(item.activity)),
            "duration": _duration(item.duration_seconds),
        })
    return rows


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
        "VERIFYING":                           "Đang xác minh danh tính",
    }.get(value, value)


def _attendance_tone(value: str) -> str:
    if value in {"CHECK-IN SUCCESS", "CHECK-OUT SUCCESS", "Attendance already complete."}:
        return "success"
    if value in {"UNKNOWN", "LIVENESS FAILED"}:
        return "danger"
    if value == "VERIFYING":
        return "info"
    if "WFH" in value:
        return "info"
    if "OFF" in value or "Chưa đăng ký" in value:
        return "neutral"
    return "warning"


def _activity_tone(value: ActivityType) -> str:
    return {
        ActivityType.AT_DESK: "success",
        ActivityType.WALKING: "info",
        ActivityType.STANDING: "neutral",
        ActivityType.DRINKING: "info",
        ActivityType.SLEEPING_SUSPECTED: "warning",
        ActivityType.AWAY: "warning",
        ActivityType.UNKNOWN: "neutral",
    }[value]


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
