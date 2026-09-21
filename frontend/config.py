"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
import ast
import urllib.parse
from dataclasses import dataclass
from datetime import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_environment(path: Path) -> None:
    """Use python-dotenv when available, with a small stdlib fallback for diagnostics/tests."""
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
        return
    except ImportError:
        # Core modules can still run before optional dependencies are installed.
        # The fallback below supports the simple KEY=VALUE format used by this project.
        load_dotenv = None  # type: ignore[assignment, misc]
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_environment(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _time(name: str, default: str) -> time:
    try:
        return time.fromisoformat(os.getenv(name, default))
    except ValueError:
        return time.fromisoformat(default)


def _parse_camera_source(value: str) -> int | str:
    value = value.strip()
    return int(value) if value.lstrip("-").isdigit() else value


def _camera_sources() -> tuple[tuple[str, int | str], ...]:
    """Load camera sources without executing an external Python configuration file."""
    explicit = os.getenv("CAMERA_SOURCE", "").strip()
    if explicit:
        return ((os.getenv("CAMERA_NAME", "Camera chính"), _parse_camera_source(explicit)),)
    config_file = os.getenv("CAMERA_CONFIG_FILE", "").strip()
    if config_file:
        path = Path(config_file).expanduser()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            values: dict[str, object] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name in {"USERNAME", "RAW_PASSWORD", "IP", "PORT"}:
                        values[name] = ast.literal_eval(node.value)
            required = {"USERNAME", "RAW_PASSWORD", "IP", "PORT"}
            if required.issubset(values):
                user = urllib.parse.quote(str(values["USERNAME"]), safe="")
                password = urllib.parse.quote(str(values["RAW_PASSWORD"]), safe="")
                host, port = values["IP"], values["PORT"]
                base = f"rtsp://{user}:{password}@{host}:{port}/cam/realmonitor"
                return (
                    ("Camera 1 · Kênh chính", f"{base}?channel=1&subtype=1"),
                    ("Camera 2 · Kênh phụ", f"{base}?channel=2&subtype=1"),
                )
        except (OSError, SyntaxError, ValueError):
            # Never silently open the laptop webcam when an RTSP config was requested.
            # An invalid string source makes CameraManager report a clear connection error.
            return (("Lỗi cấu hình camera RTSP", f"invalid-camera-config:{path}"),)
    return (("Webcam", 0),)


_CONFIGURED_CAMERAS = _camera_sources()


def _registration_camera_source() -> int | str:
    value = os.getenv("REGISTRATION_CAMERA_SOURCE", "").strip()
    return _parse_camera_source(value) if value else _CONFIGURED_CAMERAS[0][1]


@dataclass(frozen=True)
class Settings:
    base_dir: Path = BASE_DIR
    database_path: Path = BASE_DIR / "data" / "database.db"
    face_data_dir: Path = BASE_DIR / "data" / "faces"
    log_path: Path = BASE_DIR / "logs" / "app.log"
    camera_sources: tuple[tuple[str, int | str], ...] = _CONFIGURED_CAMERAS
    camera_source: int | str = _CONFIGURED_CAMERAS[0][1]
    registration_camera_source: int | str = _registration_camera_source()
    camera_reconnect_delay: int = _int("CAMERA_RECONNECT_DELAY", 3)
    camera_width: int = _int("CAMERA_WIDTH", 1280)
    camera_height: int = _int("CAMERA_HEIGHT", 720)
    rtsp_transport: str = os.getenv("RTSP_TRANSPORT", "tcp").strip().lower()
    face_threshold: float = _float("FACE_THRESHOLD", 0.50)
    process_every_n_frames: int = max(1, _int("PROCESS_EVERY_N_FRAMES", 3))
    recognition_interval: float = max(0.2, _float("RECOGNITION_INTERVAL", 0.65))
    detection_size: int = _int("DETECTION_SIZE", 640)
    min_face_size: int = _int("MIN_FACE_SIZE", 80)
    blur_threshold: float = _float("BLUR_THRESHOLD", 12.0)
    registration_samples: int = _int("REGISTRATION_SAMPLES", 12)
    registration_interval: float = _float("REGISTRATION_INTERVAL", 0.35)
    enable_liveness: bool = _bool("ENABLE_LIVENESS", False)
    attendance_cooldown: int = _int("ATTENDANCE_COOLDOWN", 60)
    checkout_min_minutes: int = _int("CHECKOUT_MIN_MINUTES", 240)
    work_start_time: time = _time("WORK_START_TIME", "09:00")
    morning_end_time: time = _time("MORNING_END_TIME", "12:00")
    afternoon_start_time: time = _time("AFTERNOON_START_TIME", "14:00")
    work_end_time: time = _time("WORK_END_TIME", "17:30")
    late_threshold: time = _time("LATE_THRESHOLD", "09:15")
    google_sheet_id: str = os.getenv("GOOGLE_SHEET_ID", "").strip()
    google_credentials_path: str = os.getenv("GOOGLE_CREDENTIALS_PATH", "").strip()
    google_worksheet: str = os.getenv("GOOGLE_WORKSHEET", "Attendance").strip()
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    # When set, Live Attendance becomes a view of the shared Camera OJT
    # backend instead of opening the RTSP cameras a second time.
    tracking_backend_url: str = os.getenv("TRACKING_BACKEND_URL", "").strip().rstrip("/")

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.face_data_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
