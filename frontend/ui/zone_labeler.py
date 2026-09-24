"""Administrator UI for drawing camera polygons in normalized image space."""
from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from auth.permissions import ADMIN, require_permission
from components.zone_polygon_editor import zone_polygon_editor
from config import settings
from ui.components import page_header


CONFIG_VERSION = 1
ZONE_FILE = settings.base_dir / "data" / "zone_labels.json"
BACKGROUND_DIR = settings.base_dir / "data" / "zone_backgrounds"
REFERENCE_DIR = settings.base_dir / "static" / "zone_samples"
REFERENCE_IMAGES = {
    "A": REFERENCE_DIR / "camera_a_reference.jpg",
    "B": REFERENCE_DIR / "camera_b_reference.jpg",
}
ZONE_KINDS = {
    "door_inside",
    "door_outside",
    "be_xinh",
    "water",
    "restroom",
    "workspace",
    "custom",
}
_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9_]{0,49}$")


def render(role: str) -> None:
    require_permission(role, "zone.configure")
    if str(role).upper() != ADMIN:
        st.error("Chỉ Quản trị viên được thay đổi vùng camera.")
        return

    page_header(
        "Gắn nhãn vùng camera",
        "Vẽ polygon trực tiếp · tự lưu khi hoàn thành vùng · khôi phục nét vẽ dở sau khi tải lại trang",
        status="Polygon Labeler",
    )
    st.info(
        "Điểm neo của người là giữa cạnh dưới bounding box (vị trí bàn chân). "
        "Vì vậy hãy vẽ vùng trên phần sàn nơi người sẽ đứng, không khoanh quanh tường hoặc toàn bộ cánh cửa.",
        icon=":material/info:",
    )

    config = _session_config()
    camera = st.segmented_control(
        "Chọn camera",
        ["A", "B"],
        default="A",
        key="zone_label_camera",
        width="stretch",
    ) or "A"
    camera_config = config["cameras"][camera]

    source_col, upload_col, capture_col = st.columns([1.2, 1.4, 1])
    saved_image = _background_path(camera).is_file()
    source_options = ["Ảnh chụp đã lưu", "Ảnh tham chiếu"] if saved_image else ["Ảnh tham chiếu"]
    with source_col:
        source = st.radio(
            "Ảnh nền",
            source_options,
            horizontal=True,
            key=f"zone_source_{camera}",
        )
    with upload_col:
        uploaded = st.file_uploader(
            "Thay ảnh nền",
            type=("jpg", "jpeg", "png", "webp"),
            key=f"zone_upload_{camera}",
            help="Nên dùng ảnh trực tiếp từ luồng camera, không có khung nhận diện.",
        )
    with capture_col:
        st.write("")
        st.write("")
        if st.button(
            "Chụp từ camera",
            icon=":material/photo_camera:",
            width="stretch",
            key=f"zone_capture_{camera}",
        ):
            try:
                frame = _fetch_mjpeg_frame(_stream_url(camera))
                normalized, size = _normalize_image(frame)
                _save_background(camera, normalized)
            except (OSError, ValueError, HTTPError, URLError) as error:
                st.error(f"Không chụp được Camera {camera}: {error}")
            else:
                camera_config["image_size"] = list(size)
                st.success(f"Đã chụp Camera {camera} · {size[0]}×{size[1]} px")
                st.rerun()

    if uploaded is not None:
        upload_key = f"{camera}:{uploaded.name}:{uploaded.size}"
        if st.session_state.get("zone_last_upload") != upload_key:
            try:
                normalized, size = _normalize_image(uploaded.getvalue())
                _save_background(camera, normalized)
            except (OSError, ValueError, UnidentifiedImageError) as error:
                st.error(f"Ảnh không hợp lệ: {error}")
            else:
                camera_config["image_size"] = list(size)
                st.session_state.zone_last_upload = upload_key
                st.toast(f"Đã cập nhật ảnh nền Camera {camera}")
                st.rerun()

    try:
        image_bytes, image_size, is_reference = _editor_image(camera, source)
    except (OSError, ValueError, UnidentifiedImageError) as error:
        st.error(f"Không đọc được ảnh Camera {camera}: {error}")
        return

    camera_config["image_size"] = list(image_size)
    if is_reference:
        st.warning(
            "Đang dùng ảnh tham chiếu từ ứng dụng IMOU. Hãy bấm “Chụp từ camera” trước khi chốt polygon "
            "để loại bỏ joystick và hình Picture-in-Picture.",
            icon=":material/warning:",
        )

    result = zone_polygon_editor(
        image_data_url=_data_url(image_bytes),
        zones=camera_config["zones"],
        camera=camera,
        key=f"zone_polygon_editor_{camera}_{'reference' if is_reference else 'saved'}",
    )
    if result and result.get("camera") == camera:
        revision = result.get("revision")
        revision_key = f"zone_editor_revision_{camera}"
        if revision != st.session_state.get(revision_key):
            try:
                _apply_editor_result(config, camera, result)
            except (OSError, ValueError) as error:
                st.error(f"Không tự lưu được polygon: {error}")
            else:
                st.session_state[revision_key] = revision
                st.toast("Đã tự động lưu polygon xuống ổ đĩa", icon="💾")

    zones = camera_config["zones"]
    st.caption(
        "Tự động lưu đang bật: bấm **Lưu vùng** là JSON được ghi ngay. "
        "Nếu đang vẽ dở, bạn có thể tải lại trang và tiếp tục từ các điểm gần nhất."
    )
    left, middle, right = st.columns([1, 1, 1.2])
    left.metric("Camera đang chỉnh", camera)
    middle.metric("Số vùng", len(zones))
    audio_ready = sum(zone["kind"] in ZONE_KINDS - {"workspace", "custom"} for zone in zones)
    right.metric("Vùng Audio Event", audio_ready)

    validation = _configuration_notes(config)
    if validation:
        with st.expander("Kiểm tra cấu hình", expanded=True):
            for tone, message in validation:
                getattr(st, tone)(message)

    save_col, export_col = st.columns([1, 1])
    with save_col:
        if st.button(
            "Lưu cấu hình polygon",
            type="primary",
            icon=":material/save:",
            width="stretch",
        ):
            try:
                _save_config(config)
            except OSError as error:
                st.error(f"Không lưu được cấu hình: {error}")
            else:
                st.success(f"Đã lưu {ZONE_FILE}")
    with export_col:
        st.download_button(
            "Tải zone_labels.json",
            data=_config_json(config),
            file_name="zone_labels.json",
            mime="application/json",
            icon=":material/download:",
            width="stretch",
        )

    with st.expander("Ý nghĩa các loại vùng"):
        st.markdown(
            """
- `door_inside` / `door_outside`: tạo sự kiện vào–ra khi có hướng di chuyển phù hợp.
- `be_xinh`: Bé Xinh phản hồi khi người ở gần đủ thời gian, không còn chào chỉ vì thấy mặt.
- `water`: nhắc uống nước theo dwell và cooldown.
- `restroom`: luôn bật chế độ riêng tư, không đọc tên người.
- `workspace`: chỉ phục vụ phân tích làm việc; không tự phát âm thanh.
- `custom`: lưu polygon nhưng cần ánh xạ sự kiện riêng trước khi dùng.
"""
        )


def _empty_config() -> dict[str, Any]:
    return {
        "version": CONFIG_VERSION,
        "coordinate_space": "normalized_image",
        "anchor": "bbox_bottom_center",
        "cameras": {
            "A": {"image_size": [0, 0], "zones": []},
            "B": {"image_size": [0, 0], "zones": []},
        },
    }


def _session_config() -> dict[str, Any]:
    if "zone_label_config" not in st.session_state:
        st.session_state.zone_label_config = _load_config(ZONE_FILE)
    return st.session_state.zone_label_config


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _empty_config()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("root must be an object")
        config = _empty_config()
        cameras = payload.get("cameras", {})
        for camera in ("A", "B"):
            item = cameras.get(camera, {}) if isinstance(cameras, dict) else {}
            config["cameras"][camera]["zones"] = _validate_zones(item.get("zones", []))
            size = item.get("image_size", [0, 0])
            if isinstance(size, list) and len(size) == 2:
                config["cameras"][camera]["image_size"] = [int(size[0]), int(size[1])]
        return config
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return _empty_config()


def _validate_zones(raw_zones: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_zones, list):
        raise ValueError("zones phải là một danh sách")
    zones: list[dict[str, Any]] = []
    used: set[str] = set()
    for raw in raw_zones:
        if not isinstance(raw, dict):
            raise ValueError("mỗi zone phải là object")
        zone_id = str(raw.get("id", "")).strip().lower()
        label = " ".join(str(raw.get("label", "")).split())[:60]
        kind = str(raw.get("kind", "custom")).strip().lower()
        color = str(raw.get("color", "#22c55e")).strip()
        if not _ID.fullmatch(zone_id) or zone_id in used:
            raise ValueError(f"mã vùng trống, trùng hoặc không hợp lệ: {zone_id!r}")
        if not label:
            raise ValueError(f"vùng {zone_id!r} chưa có tên")
        if kind not in ZONE_KINDS:
            raise ValueError(f"loại vùng không được hỗ trợ: {kind!r}")
        if not _COLOR.fullmatch(color):
            raise ValueError(f"màu vùng không hợp lệ: {color!r}")
        points = raw.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(f"vùng {zone_id!r} cần ít nhất 3 điểm")
        normalized: list[list[float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"tọa độ trong {zone_id!r} không hợp lệ")
            x, y = float(point[0]), float(point[1])
            if not 0 <= x <= 1 or not 0 <= y <= 1:
                raise ValueError(f"tọa độ trong {zone_id!r} nằm ngoài 0–1")
            normalized.append([round(x, 6), round(y, 6)])
        if _polygon_area(normalized) < 0.00005:
            raise ValueError(f"vùng {zone_id!r} có diện tích quá nhỏ")
        used.add(zone_id)
        zones.append({
            "id": zone_id,
            "label": label,
            "kind": kind,
            "points": normalized,
            "color": color.lower(),
            "privacy": "anonymous_audio" if kind == "restroom" or raw.get("privacy") == "anonymous_audio" else "standard",
            "enabled": bool(raw.get("enabled", True)),
        })
    return zones


def _polygon_area(points: list[list[float]]) -> float:
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2


def _configuration_notes(config: dict[str, Any]) -> list[tuple[str, str]]:
    zones = [
        (camera, zone)
        for camera, item in config["cameras"].items()
        for zone in item["zones"]
    ]
    notes: list[tuple[str, str]] = []
    if not zones:
        return [("info", "Chưa có polygon. Hãy bắt đầu bằng Camera B cho vùng cửa, sau đó Camera A cho vùng Bé Xinh/khu làm việc.")]
    kinds = {zone["kind"] for _, zone in zones}
    if "door_inside" in kinds and "door_outside" not in kinds:
        notes.append(("warning", "Đã có door_inside nhưng chưa có door_outside; việc suy ra hướng vào–ra có thể thiếu ổn định."))
    if "door_outside" in kinds and "door_inside" not in kinds:
        notes.append(("warning", "Đã có door_outside nhưng chưa có door_inside; nên vẽ cặp vùng hai phía cửa."))
    if "restroom" in kinds:
        notes.append(("success", "Vùng WC đã được ép privacy=anonymous_audio: Bé Xinh không đọc tên người."))
    if "custom" in kinds:
        notes.append(("warning", "Vùng custom được lưu nhưng chưa tự sinh Audio Event cho tới khi có ánh xạ nghiệp vụ."))
    if not notes:
        notes.append(("success", "Các polygon hiện tại hợp lệ về tên, tọa độ, diện tích và quy tắc riêng tư."))
    return notes


def _config_json(config: dict[str, Any]) -> str:
    payload = deepcopy(config)
    payload["version"] = CONFIG_VERSION
    payload["coordinate_space"] = "normalized_image"
    payload["anchor"] = "bbox_bottom_center"
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _apply_editor_result(
    config: dict[str, Any],
    camera: str,
    result: dict[str, Any],
    path: Path = ZONE_FILE,
) -> list[dict[str, Any]]:
    """Validate and durably persist one component update before acknowledging it."""
    if camera not in {"A", "B"} or result.get("camera") != camera:
        raise ValueError("camera trả về không khớp")
    zones = _validate_zones(result.get("zones"))
    config["cameras"][camera]["zones"] = zones
    _save_config(config, path)
    return zones


def _save_config(config: dict[str, Any], path: Path = ZONE_FILE) -> None:
    for camera in ("A", "B"):
        config["cameras"][camera]["zones"] = _validate_zones(
            config["cameras"][camera]["zones"]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_config_json(config), encoding="utf-8")
    temporary.replace(path)


def _background_path(camera: str) -> Path:
    return BACKGROUND_DIR / f"camera_{camera.lower()}.jpg"


def _save_background(camera: str, content: bytes) -> None:
    path = _background_path(camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _editor_image(camera: str, source: str) -> tuple[bytes, tuple[int, int], bool]:
    saved = _background_path(camera)
    if source == "Ảnh chụp đã lưu" and saved.is_file():
        content, size = _normalize_image(saved.read_bytes())
        return content, size, False
    reference = REFERENCE_IMAGES[camera]
    content, size = _normalize_image(reference.read_bytes(), crop_imou_reference=True)
    return content, size, True


def _normalize_image(
    content: bytes,
    *,
    crop_imou_reference: bool = False,
) -> tuple[bytes, tuple[int, int]]:
    if len(content) > 15 * 1024 * 1024:
        raise ValueError("ảnh lớn hơn giới hạn 15 MB")
    with Image.open(BytesIO(content)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if crop_imou_reference:
            # IMOU mobile screenshots place a centered 16:9 camera frame
            # inside a wider canvas.  Crop by aspect ratio instead of one
            # exact screenshot width (Windows/browser capture may remove a
            # few edge pixels, e.g. 1908 rather than 1920).
            target_width = math.ceil(image.height * 16 / 9)
            if image.width - target_width >= 40:
                left = (image.width - target_width) // 2
                image = image.crop((left, 0, left + target_width, image.height))
        if image.width < 320 or image.height < 180:
            raise ValueError("ảnh quá nhỏ; tối thiểu 320×180 px")
        output = BytesIO()
        image.save(output, format="JPEG", quality=90, optimize=True)
        return output.getvalue(), image.size


def _data_url(content: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(content).decode("ascii")


def _stream_url(camera: str) -> str:
    base = settings.tracking_backend_url or "http://127.0.0.1:8765"
    return f"{base.rstrip('/')}/cam_{camera.lower()}.mjpg"


def _fetch_mjpeg_frame(url: str, max_bytes: int = 5 * 1024 * 1024) -> bytes:
    request = Request(url, headers={"User-Agent": "AI-Mind-Zone-Labeler/1.0"})
    buffer = bytearray()
    with urlopen(request, timeout=8) as response:
        while len(buffer) < max_bytes:
            chunk = response.read(min(65536, max_bytes - len(buffer)))
            if not chunk:
                break
            buffer.extend(chunk)
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
            if start >= 0 and end >= 0:
                return bytes(buffer[start:end + 2])
    raise ValueError("không tìm thấy khung JPEG trong luồng MJPEG")


__all__ = [
    "_apply_editor_result",
    "_configuration_notes",
    "_empty_config",
    "_load_config",
    "_normalize_image",
    "_save_config",
    "_validate_zones",
    "render",
]
