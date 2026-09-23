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
        st.caption(f"Chờ đồng bộ: {health['pending_events']} sự kiện · "
                   f"{health['pending_records']} bản ghi · {health['conflicts']} xung đột")
        with st.expander("Sự kiện cần xác nhận danh tính"):
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
        with st.expander("Xung đột đồng bộ cloud"):
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
    except (RuntimeError, ValueError) as error:
        st.warning(str(error))
    client = CameraOjtClient(settings.camera_ojt_url, settings.camera_ojt_supervisor_url)
    try:
        status = client.status()
        online = True
    except RuntimeError as error:
        status, online = {}, False
        backend_error = str(error)

    page_header(
        "Hệ thống AI & Trợ lý",
        "Trạng thái camera-ojt, đồng bộ SQLite, voice Hà Linh và tìm kiếm Gemini/Tavily",
        status="Backend trực tuyến" if online else "Backend ngoại tuyến",
    )

    services = status.get("services") if isinstance(status.get("services"), dict) else {}
    cols = st.columns(4)
    values = (
        ("Camera backend", "Hoạt động" if online else "Mất kết nối", online),
        ("Bridge SQLite", "Đang chạy" if getattr(bridge, "running", False) else "Đang dừng", bool(getattr(bridge, "running", False))),
        ("Voice Hà Linh", "Đã bật" if services.get("voice_enabled") else "Chưa bật", bool(services.get("voice_enabled"))),
        ("Gemini / Search", (services.get("search_provider") or "Chưa cấu hình") if services.get("gemini_configured") else "Thiếu Gemini key", bool(services.get("gemini_configured"))),
    )
    for column, (label, value, ok) in zip(cols, values):
        with column:
            st.markdown(f"**{label}**")
            st.markdown("✅ " + value if ok else "⚠️ " + value)

    if not online:
        st.error(backend_error)
        st.code("python3 scripts/run_workstate.py --stream-host 0.0.0.0")
    bridge_error = getattr(bridge, "error", "")
    if bridge_error:
        st.warning(f"Bridge: {bridge_error}")

    section_header("Hỏi Hà Linh", "Câu hỏi được xử lý tại backend; API key không đi qua trình duyệt")
    with st.form("assistant_question", clear_on_submit=False):
        question = st.text_input(
            "Câu hỏi", placeholder="Ví dụ: Giá Bitcoin hôm nay hoặc hiện có bao nhiêu người trong phòng?",
        )
        submitted = st.form_submit_button(
            "Gửi câu hỏi", type="primary", disabled=not online,
            icon=":material/send:",
        )
    if submitted:
        try:
            with st.spinner("Hà Linh đang xử lý…"):
                result = client.ask(question)
            if result.get("ok"):
                st.success(str(result.get("answer") or "Không có câu trả lời."))
                tools = result.get("tools") or []
                st.caption(f"Công cụ: {', '.join(map(str, tools)) or 'không'} · {result.get('tool_time_s', 0)} giây")
            else:
                st.error(str(result.get("message") or "Không thể xử lý câu hỏi."))
        except RuntimeError as error:
            st.error(str(error))

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
