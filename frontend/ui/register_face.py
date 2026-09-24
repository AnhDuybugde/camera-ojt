from __future__ import annotations

from html import escape

import cv2
import streamlit as st

from auth.permissions import EMPLOYEE, require_permission
from camera.camera_manager import CameraManager
from config import settings
from database.db import Database
from face.detector import FaceDetector
from face.face_registration import FaceRegistrationService
from face.tracking_enrollment import TrackingEnrollmentClient, TrackingEnrollmentError
from face.recognizer import FaceRecognizer
from ui.components import page_header, section_header, translate_message

# Maximum number of uploaded images processed at once
_MAX_UPLOAD = 20
_ALLOWED_TYPES = ["jpg", "jpeg", "png"]


def render(
    db: Database, detector: FaceDetector, recognizer: FaceRecognizer, role: str
) -> None:
    require_permission(role, "face.register")
    page_header(
        "Đăng ký khuôn mặt",
        "Thu thập mẫu khuôn mặt chất lượng cao cho hệ thống nhận diện",
    )

    employees = db.list_employees()
    if role == EMPLOYEE:
        owner = st.session_state.get("employee_id")
        employees = [row for row in employees if owner and row["employee_id"] == owner]
        if employees and employees[0]["has_face"]:
            st.success("Bạn đã đăng ký khuôn mặt. Nếu cần đăng ký lại, hãy liên hệ quản lý.")
            return
    if not employees:
        st.info("Chưa có hồ sơ nhân viên. Hãy thêm nhân viên trước trong mục Nhân viên.")
        return

    tab_camera, tab_upload = st.tabs(
        ["📷  Camera trực tiếp", "🖼️  Tải ảnh lên"]
    )

    with tab_camera:
        _tab_camera(db, detector, recognizer, employees)

    with tab_upload:
        _tab_upload(db, detector, recognizer, employees)


# ── Tab 1: Camera trực tiếp (unchanged logic) ────────────────────────────────

def _tab_camera(
    db: Database,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    employees: list[dict],
) -> None:
    available = [row for row in employees if not row["has_face"]]
    if not available:
        st.success("Tất cả nhân viên đã có dữ liệu khuôn mặt.")
        return

    left, guide = st.columns([1.55, 1], gap="large")
    with left:
        with st.container(border=True):
            section_header("Thông tin đăng ký",
                           f"Cần {settings.registration_samples} mẫu hợp lệ")
            labels = {
                f'{row["employee_id"]} · {row["full_name"]}': row["employee_id"]
                for row in available
            }
            selected = st.selectbox("Nhân viên", list(labels), key="cam_employee")
            camera_options = dict(settings.camera_sources)
            if settings.registration_camera_source not in camera_options.values():
                camera_options["Webcam laptop"] = settings.registration_camera_source
            selected_camera = st.selectbox(
                "Nguồn camera đăng ký",
                list(camera_options),
                help="Camera thực tế được lấy từ cấu hình hiện tại của hệ thống.",
                key="cam_camera_choice",
            )
            st.caption(
                "Nhìn thẳng, giữ khuôn mặt đủ sáng rồi từ từ quay nhẹ sang trái và phải."
            )
            if st.button(
                "Bắt đầu đăng ký",
                type="primary",
                width="stretch",
                icon=":material/face:",
                key="cam_register_btn",
            ):
                _register_camera(
                    labels[selected],
                    camera_options[selected_camera],
                    db, detector, recognizer,
                )

    with guide:
        section_header("Hướng dẫn chất lượng")
        tips = [
            ("Chỉ một người trong khung hình",
             "Tránh người khác xuất hiện phía sau."),
            ("Ánh sáng đều trên khuôn mặt",
             "Không đứng ngược sáng hoặc quá tối."),
            ("Giữ ổn định và đổi góc nhẹ",
             "Hệ thống tự thu đủ mẫu hợp lệ."),
        ]
        items = "".join(
            f'<div class="stitch-timeline-event" style="padding-bottom:12px;">'
            f'  <span class="stitch-timeline-dot stitch-timeline-dot-ok">'
            f'    <i class="bi bi-check-lg" style="font-size:9px;"></i>'
            f'  </span>'
            f'  <div>'
            f'    <div class="stitch-timeline-name">{escape(title)}</div>'
            f'    <p class="stitch-timeline-sub">{escape(sub)}</p>'
            f'  </div>'
            f'</div>'
            for title, sub in tips
        )
        st.markdown(
            f'<div class="stitch-card" style="padding:var(--sp-md);">'
            f'  <div class="stitch-timeline">{items}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _register_camera(
    employee_id: str,
    source: int | str,
    db: Database,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
) -> None:
    employee = db.get_employee(employee_id)
    if not employee:
        st.error("Không tìm thấy nhân viên cần đăng ký.")
        return
    camera = CameraManager(source).start()
    preview, bar, status = st.empty(), st.progress(0), st.empty()
    try:
        if not camera.wait_for_frame():
            raise RuntimeError(camera.error or "Camera chưa cung cấp được khung hình.")

        def update(count: int, total: int, frame, message: str) -> None:
            _msgs = {
                "No face detected. Look toward the camera and move closer.":
                    "Chưa thấy khuôn mặt. Hãy nhìn vào camera và tiến gần hơn.",
                "Captured - slowly change your head angle":
                    "Đã lấy mẫu · từ từ thay đổi góc khuôn mặt.",
                "The model could not extract a face embedding.":
                    "Không thể trích xuất đặc trưng khuôn mặt.",
            }
            bar.progress(count / total, text=f"Đã thu {count}/{total} mẫu")
            status.caption(translate_message(_msgs.get(message, message)))
            preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                          channels="RGB", width="stretch")

        FaceRegistrationService(db, detector, actor_role=st.session_state.role, actor_employee_id=st.session_state.get("employee_id")).register_from_capture(
            employee_id, camera, progress=update,
            before_save=_tracking_enrollment_callback(
                employee_id, employee["full_name"], overwrite=False
            ),
        )
        recognizer.reload()
        st.success("Đăng ký khuôn mặt thành công.")
    except Exception as exc:
        st.error(f"Đăng ký khuôn mặt thất bại: {translate_message(str(exc))}")
    finally:
        camera.stop()


# ── Tab 2: Tải ảnh lên ───────────────────────────────────────────────────────

def _tab_upload(
    db: Database,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    employees: list[dict],
) -> None:
    left, guide = st.columns([1.55, 1], gap="large")

    with left:
        with st.container(border=True):
            section_header("Đăng ký bằng ảnh", "Hỗ trợ JPG, JPEG, PNG · Tối thiểu 1 ảnh")

            # Employee selector – ALL employees (allow overwrite)
            emp_labels = {
                f'{row["employee_id"]} · {row["full_name"]}': row
                for row in employees
            }
            selected_label = st.selectbox("Nhân viên", list(emp_labels), key="up_employee")
            selected_row = emp_labels[selected_label]
            employee_id = selected_row["employee_id"]

            # Overwrite warning if face already exists
            overwrite = False
            if selected_row["has_face"]:
                st.warning(
                    f"**{selected_row['full_name']}** đã có dữ liệu khuôn mặt. "
                    "Đăng ký mới sẽ ghi đè dữ liệu cũ."
                )
                overwrite = st.checkbox(
                    "Xác nhận ghi đè dữ liệu khuôn mặt hiện tại",
                    key="up_overwrite",
                )
            else:
                overwrite = True   # no existing face → no confirmation needed

            uploaded = st.file_uploader(
                "Chọn ảnh (có thể chọn nhiều ảnh cùng lúc)",
                type=_ALLOWED_TYPES,
                accept_multiple_files=True,
                key="up_files",
                help=f"Tối đa {_MAX_UPLOAD} ảnh mỗi lần. Mỗi ảnh phải có đúng 1 khuôn mặt rõ ràng.",
            )
            if uploaded and len(uploaded) > _MAX_UPLOAD:
                st.warning(f"Chỉ xử lý {_MAX_UPLOAD} ảnh đầu tiên (giới hạn mỗi lần).")
                uploaded = uploaded[:_MAX_UPLOAD]

            btn_disabled = not uploaded or not overwrite
            if st.button(
                "Đăng ký khuôn mặt",
                type="primary",
                width="stretch",
                icon=":material/upload:",
                disabled=btn_disabled,
                key="up_register_btn",
            ):
                _register_upload(
                    employee_id,
                    selected_row["full_name"],
                    [f.read() for f in uploaded],
                    db, detector, recognizer,
                    overwrite=overwrite,
                )

    with guide:
        section_header("Yêu cầu ảnh hợp lệ")
        reqs = [
            ("Đúng 1 khuôn mặt trong ảnh",    "Không chấp nhận ảnh nhóm hoặc ảnh không có mặt."),
            ("Ảnh rõ nét, đủ sáng",            "Tránh ảnh mờ, ngược sáng, hoặc quá tối."),
            ("Khuôn mặt đủ lớn",               "Khuôn mặt nên chiếm ít nhất 1/5 chiều cao ảnh."),
            ("Định dạng JPG / JPEG / PNG",      "File ảnh không bị hỏng hoặc mã hóa đặc biệt."),
        ]
        items = "".join(
            f'<div class="stitch-timeline-event" style="padding-bottom:12px;">'
            f'  <span class="stitch-timeline-dot stitch-timeline-dot-ok">'
            f'    <i class="bi bi-check-lg" style="font-size:9px;"></i>'
            f'  </span>'
            f'  <div>'
            f'    <div class="stitch-timeline-name">{escape(title)}</div>'
            f'    <p class="stitch-timeline-sub">{escape(sub)}</p>'
            f'  </div>'
            f'</div>'
            for title, sub in reqs
        )
        st.markdown(
            f'<div class="stitch-card" style="padding:var(--sp-md);">'
            f'  <div class="stitch-timeline">{items}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Image previews
        if uploaded:
            st.markdown(
                '<div class="stitch-section-head" style="margin-top:var(--sp-md);">'
                '  <div class="stitch-section-title">Ảnh đã chọn</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            _render_image_previews(uploaded)


def _register_upload(
    employee_id: str,
    full_name: str,
    image_bytes_list: list[bytes],
    db: Database,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    overwrite: bool = True,
) -> None:
    service = FaceRegistrationService(db, detector, actor_role=st.session_state.role, actor_employee_id=st.session_state.get("employee_id"))
    with st.spinner("Đang phân tích ảnh và trích xuất khuôn mặt…"):
        try:
            result = service.register_from_images(
                employee_id, image_bytes_list, overwrite=overwrite,
                before_save=_tracking_enrollment_callback(
                    employee_id, full_name, overwrite=overwrite
                ),
            )
        except TrackingEnrollmentError as exc:
            st.error(f"Không thể đồng bộ khuôn mặt với Camera OJT: {exc}")
            return
        except ValueError as exc:
            msg = str(exc)
            if msg == "ALREADY_HAS_FACE":
                st.error("Nhân viên đã có dữ liệu khuôn mặt. Hãy tích xác nhận ghi đè.")
            else:
                st.error(translate_message(msg))
            return
        except Exception as exc:
            st.error(f"Lỗi không xác định: {exc}")
            return

    if result["success"]:
        recognizer.reload()
        valid, rejected = result["valid_count"], result["rejected_count"]
        st.success(
            f"**Đăng ký khuôn mặt thành công** cho **{full_name}** ({employee_id})  \n"
            f"Ảnh hợp lệ: **{valid}** · Ảnh bị loại: **{rejected}**"
        )
        if result["rejected_reasons"]:
            with st.expander(f"Xem lý do loại {rejected} ảnh"):
                for i, reason in enumerate(result["rejected_reasons"], 1):
                    st.markdown(f"- Ảnh #{i}: {reason}")
    else:
        st.error(result["message"])
        if result["rejected_reasons"]:
            with st.expander("Xem chi tiết lỗi từng ảnh"):
                for i, reason in enumerate(result["rejected_reasons"], 1):
                    st.markdown(f"- Ảnh #{i}: {reason}")


def _tracking_enrollment_callback(
    employee_id: str, full_name: str, *, overwrite: bool
):
    if not settings.tracking_backend_url:
        return None
    client = TrackingEnrollmentClient(settings.tracking_backend_url)

    def sync(image_bytes: bytes) -> None:
        client.register(
            employee_id, full_name, image_bytes, overwrite=overwrite
        )

    return sync


def _render_image_previews(uploaded_files: list) -> None:
    """Show thumbnails in a 3-column grid with filename caption."""
    cols = st.columns(3)
    for idx, file in enumerate(uploaded_files):
        with cols[idx % 3]:
            file.seek(0)
            st.image(file.read(), width=120)
            name = file.name
            st.caption(f"{'…' + name[-14:] if len(name) > 16 else name}")
