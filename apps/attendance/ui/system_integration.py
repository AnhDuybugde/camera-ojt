"""Admin operations page for camera-ojt, voice and web search."""
from __future__ import annotations

import streamlit as st

from auth.permissions import require_permission
from config import settings
from integration.camera_ojt import CameraOjtClient
from ui.components import page_header, section_header, status_badge, TableColumn, data_table


def render(role: str, bridge: object | None = None) -> None:
    require_permission(role, "settings.view")
    from integration.backend import RemoteService
    operations = RemoteService("operations")
    try:
        health = operations.diagnostics()
        backend_online = True
    except (RuntimeError, ValueError) as error:
        health, backend_online = {}, False
        backend_error = str(error)
    try:
        ai_status = operations.ai_status() if backend_online else {}
    except (RuntimeError, ValueError):
        ai_status = {}
    client = CameraOjtClient(settings.camera_ojt_url, settings.camera_ojt_supervisor_url)
    try:
        status = client.status()
        pipeline_online = True
    except RuntimeError as error:
        status, pipeline_online = {}, False
        pipeline_error = str(error)

    page_header(
        "Hệ thống AI & Trợ lý",
        "Camera · Voice Hà Linh · Điểm danh · Event log — backend :8767 + pipeline :8765",
        status="Pipeline trực tuyến" if pipeline_online else "Pipeline ngoại tuyến",
    )
    if backend_online:
        st.caption(f"Backend trực tuyến — chờ đồng bộ: {health.get('pending_events', '?')} sự kiện · "
                   f"{health.get('pending_records', '?')} bản ghi · {health.get('conflicts', '?')} xung đột · "
                   f"{health.get('review_events', '?')} sự kiện cần duyệt")
    else:
        st.warning(backend_error if 'backend_error' in locals() else "Backend chưa chạy. Khởi động bằng camera-ojt run.")

    services = status.get("services") if isinstance(status.get("services"), dict) else {}
    gemini_ok = bool(services.get("gemini_configured") or ai_status.get("gemini_configured"))
    provider = services.get("search_provider") or ai_status.get("search_provider") or "Chưa cấu hình"
    cols = st.columns(4)
    values = (
        ("1 · Camera", "Hoạt động" if pipeline_online else "Mất kết nối", pipeline_online),
        ("2 · Voice Hà Linh", "Đã bật" if (services.get("voice_enabled") or ai_status.get("voice_enabled")) else "Chưa bật",
         bool(services.get("voice_enabled") or ai_status.get("voice_enabled"))),
        ("3 · Suy nghĩ Gemini", provider if gemini_ok else "Thiếu Gemini key", gemini_ok),
        ("4 · Supabase", "Đã cấu hình" if ai_status.get("supabase_configured") else "Chưa cấu hình / local-only",
         bool(ai_status.get("supabase_configured"))),
    )
    for column, (label, value, ok) in zip(cols, values):
        with column:
            st.markdown(f"**{label}**")
            st.markdown("✅ " + value if ok else "⚠️ " + value)

    # ── 1. Camera ──
    section_header("1 · Camera trực tiếp", "Pipeline :8765 — R1/R2/R3 sơ khai, tune sau")
    if not pipeline_online:
        st.error(pipeline_error if 'pipeline_error' in locals() else "Pipeline ngoại tuyến")
        st.code("camera-ojt run  # hoặc: camera-ojt run --profile webcam --device cpu")
    else:
        cameras = status.get("cameras") if isinstance(status.get("cameras"), list) else []
        st.caption(f"{len(cameras)} kênh · count A/B: {status.get('count_a', '?')}/{status.get('count_b', '?')} · "
                   f"khách chưa định danh: {status.get('anonymous_count', 0)}")

    # ── 2. Voice + suy nghĩ ──
    section_header("2 · Voice & Hỏi Hà Linh", "Pipeline trước, backend :8767 fallback khi tắt camera")
    with st.form("assistant_question", clear_on_submit=False):
        question = st.text_input(
            "Câu hỏi", placeholder="Ví dụ: Giá Bitcoin hôm nay hoặc hiện có bao nhiêu người trong phòng?",
        )
        submitted = st.form_submit_button(
            "Gửi câu hỏi", type="primary", disabled=not (pipeline_online or backend_online),
            icon=":material/send:",
        )
    if submitted:
        result = None
        if pipeline_online:
            try:
                with st.spinner("Hà Linh đang xử lý…"):
                    result = client.ask(question)
            except RuntimeError:
                result = None
        if result is None and backend_online:
            try:
                with st.spinner("Hà Linh backend đang xử lý…"):
                    result = operations.ask(question)
            except (RuntimeError, ValueError) as error:
                st.error(str(error))
                result = {"ok": False}
        if result:
            if result.get("ok"):
                st.success(str(result.get("answer") or "Không có câu trả lời."))
                tools = result.get("tools") or []
                st.caption(f"Công cụ: {', '.join(map(str, tools)) or 'không'} · {result.get('tool_time_s', 0)} giây")
            else:
                st.error(str(result.get("message") or "Không thể xử lý câu hỏi."))
    bridge_error = getattr(bridge, "error", "")
    if bridge_error:
        st.warning(f"Bridge: {bridge_error}")

    section_header("3 · Điểm danh hôm nay", "Từ pipeline realtime + SQLite backend")
    attendance_today = status.get("attendance_today") if isinstance(status.get("attendance_today"), list) else []
    data_table(
        [
            TableColumn("employee", "Nhân viên", 230),
            TableColumn("time", "Giờ vào", 110),
            TableColumn("score", "Điểm mặt", 100),
        ],
        [{
            "employee": item.get("person_name") or item.get("person_id") or "—",
            "time": str(item.get("check_in_at") or "")[11:19] or "—",
            "score": f"{float(item.get('face_score')):.1%}" if item.get("face_score") is not None else "—",
        } for item in attendance_today if isinstance(item, dict)],
        empty_message="Chưa có lượt điểm danh hôm nay. Xem chi tiết ở Lịch sử chấm công.", compact=True,
    )

    section_header("4 · Event log & Kiểm duyệt", "Sự kiện cửa + xung đột Supabase")
    if backend_online:
        with st.expander("Sự kiện cần xác nhận danh tính", expanded=False):
            events = operations.review_events()
            if not events:
                st.info("Không có sự kiện chờ duyệt.")
            else:
                selected = st.selectbox("Sự kiện", events,
                    format_func=lambda e: f"{e['observed_at']} · {e['kind']} · camera {e['camera_id']}")
                employee = st.text_input("Mã nhân viên đã xác minh")
                reason = st.text_input("Lý do / bằng chứng xác nhận")
                if st.button("Xác nhận sự kiện", disabled=not employee or len(reason.strip()) < 5):
                    outcome = operations.review(selected["event_id"], employee, reason)
                    st.success(outcome["message"])
                    st.rerun()
        with st.expander("Xung đột đồng bộ cloud", expanded=False):
            store = st.selectbox("Kho dữ liệu", ["business", "enrollment"])
            conflicts = operations.conflicts(store)
            if conflicts:
                conflict = st.selectbox("Bản ghi", conflicts,
                    format_func=lambda c: f"{c['kind']} · {c['record_key']}")
                choice = st.radio("Giữ phiên bản", ["cloud", "local"])
                st.caption("Chọn local sẽ gửi lại thay đổi; chọn cloud sẽ thay bản local bằng bản trên cloud.")
                if st.button("Áp dụng lựa chọn"):
                    operations.resolve_conflict(conflict["kind"], conflict["record_key"], choice, store)
                    st.rerun()
            else:
                st.info("Không có xung đột.")
    else:
        st.warning("Backend offline nên chưa xem được event log.")

    section_header("Dữ liệu realtime", "Output chuẩn camera-ojt.status.v1")
    people = status.get("people") if isinstance(status.get("people"), list) else []
    data_table(
        [
            TableColumn("gid", "Global ID", 90),
            TableColumn("name", "Nhân viên", 190),
            TableColumn("label", "Trạng thái", 140),
            TableColumn("camera", "Camera", 90),
        ],
        [{
            "gid": f"G{person.get('gid', '?')}",
            "name": person.get("name") or "Chưa xác định",
            "label": status_badge(str(person.get("label") or "Unknown"), "success" if person.get("in_room") else "neutral"),
            "camera": person.get("camera") or "—",
        } for person in people if isinstance(person, dict)],
        empty_message="Backend chưa phát hiện người nào.", compact=True,
    )
