"""Demo tracking Global ID 2 channel IMOU + voice Ha Linh (mic/loa camera).

Moi channel chay doc lap: YOLO(26s) -> ByteTrack (tracking ngan han:
motion prediction, IoU matching, data association, track buffer).
Mot GlobalIdentityManager dung chung cho ca 2 channel tra loi
"day la nguoi nao" (Global Person ID on dinh xuyen tracklet, xuyen mat dau
dai, xuyen channel) bang appearance gallery + cost matrix + Hungarian +
gating + lifecycle ACTIVE/TEMP_LOST/LONG_LOST/UNRESOLVED.
Lop business theo channel (ChannelBusinessTracker) tra loi rieng
"nguoi do dang lam gi" (WORKING/AWAY_TEMP/POSSIBLY_OUT/RETURNING) va khong
duoc phep anh huong toi ID.

Am thanh HOAN TOAN camera IMOU (doi lap voi run_workstate_local.py):
  * Greet vay tay/palm + face-trigger phat ra loa camera (backend imou_p2p).
  * Voice Ha Linh medium-only (scripts/halinh_assistant.py) chay subprocess
    rieng, mic RTSP camera + loa camera P2P, tu restart khi crash.
  * Da bo hello imou / STT small.

Chay:
  python scripts/run_workstate.py --config config/default.yaml --display
  python scripts/run_workstate.py --source-a 0 --source-b data/samples/hallway.mp4 --display
  python scripts/run_workstate.py --no-halinh  # tat voice Ha Linh

RTSP IMOU lay tu .env (IMOU_IP/USER/PASSWORD) neu khong truyen --source.
Nhan 'q' hoac ESC de thoat.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import subprocess
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")
# Giu console cho log nghiep vu cua pipeline. Cac thu vien C++ ben duoi
# thuong in thong tin khoi tao dai (MediaPipe/ONNX) du pipeline van binh thuong.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
warnings.filterwarnings(
    "ignore",
    message=r"Cython evaluation .* is unavailable.*",
    category=UserWarning,
)
# Low-latency RTSP: TCP transport + no buffering + low delay decode.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
)

import cv2
import numpy as np

from camera_tracking.camera import FrameHub
from camera_tracking.config import load_config
from camera_tracking.detection import YoloPersonDetector, resolve_device

try:  # CUDA/cuDNN tu pip wheels (khong can CUDA Toolkit he thong).
    from camera_tracking.face.embeddings import ensure_cuda_dlls
    ensure_cuda_dlls()
except (ImportError, OSError) as error:
    print(f"CUDA runtime discovery skipped: {error}", file=sys.stderr)
from camera_tracking.domain import Frame, Track, TrackEvent
from camera_tracking.gesture.focus import ForegroundSelector
from camera_tracking.runtime import StageMetrics
from camera_tracking.tracking import (
    ByteTrackTracker,
    DailyIdentityStore,
    GlobalIdentityConfig,
    GlobalIdentityManager,
    StablePersonCount,
)
from camera_tracking.visualization import (
    draw_global_labels,
    draw_hand_landmarks,
    draw_person_tracks,
)
from camera_tracking.workstate import (
    LABEL_UNKNOWN,
    ChannelBusinessTracker,
    IdentityReconciler,
    OsnetEmbedding,
    RoomPresenceAggregator,
    WorkstateConsumer,
    WorkstationZone,
)

reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
if reconfigure_stdout is not None:
    try:
        reconfigure_stdout(encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"Khong the dat UTF-8 cho terminal: {error}", file=sys.stderr)


def normalize_source(src) -> int | str:
    """'0' -> 0 (webcam), '1' -> 1, con lai giu nguyen (file/RTSP)."""
    if isinstance(src, int):
        return src
    if isinstance(src, str):
        s = src.strip()
        if s.isdigit():
            return int(s)
        return s
    return src


def _substream_fallback(source: str) -> str | None:
    parts = urlsplit(source)
    query = parse_qsl(parts.query, keep_blank_values=True)
    if not any(key == "subtype" and value == "0" for key, value in query):
        return None
    fallback_query = [
        (key, "1" if key == "subtype" else value) for key, value in query
    ]
    return urlunsplit(parts._replace(query=urlencode(fallback_query)))


def open_capture(
    source,
    attempts: int = 3,
    retry_delay_s: float = 1.0,
    camera_name: str | None = None,
) -> cv2.VideoCapture:
    source = normalize_source(source)
    if isinstance(source, int):
        # Webcam tren Windows can DSHOW, khong dung FFMPEG.
        if os.name == "nt":
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.open(source)
        return cap

    source_text = str(source)
    is_rtsp = source_text.lower().startswith("rtsp://")
    max_attempts = max(1, attempts) if is_rtsp else 1
    candidates = [source_text]
    fallback = _substream_fallback(source_text) if is_rtsp else None
    if fallback is not None:
        candidates.append(fallback)

    cap = cv2.VideoCapture()
    open_params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        20000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        15000,
    ]
    for candidate_index, candidate in enumerate(candidates):
        if candidate_index == 1:
            prefix = f"Camera {camera_name}: " if camera_name else ""
            print(f"{prefix}main stream unavailable, trying sub stream.")
        for attempt in range(1, max_attempts + 1):
            try:
                cap.open(candidate, cv2.CAP_FFMPEG, open_params)
            except cv2.error:
                cap.open(candidate, cv2.CAP_FFMPEG)
            if cap.isOpened():
                if is_rtsp:
                    # Drop buffered old frames: always decode the newest one.
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except cv2.error:
                        pass
                return cap
            cap.release()
            if attempt < max_attempts:
                time.sleep(max(0.0, retry_delay_s) * attempt)
    return cap


@dataclass
class ResilientCapture:
    source: int | str
    name: str
    capture: cv2.VideoCapture
    open_attempts: int
    max_reconnects: int = 3
    failure_threshold: int = 5
    failure_count: int = 0
    reconnect_count: int = 0
    exhausted: bool = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = False, None
        if self.capture.isOpened():
            try:
                ok, frame = self.capture.read()
            except cv2.error:
                ok, frame = False, None
        if ok and frame is not None:
            self.failure_count = 0
            return True, frame

        self.failure_count += 1
        if self.failure_count < max(1, self.failure_threshold):
            return False, None
        if self.reconnect_count >= max(0, self.max_reconnects):
            self.exhausted = True
            return False, None

        self.reconnect_count += 1
        self.failure_count = 0
        self.capture.release()
        print(
            f"Camera {self.name}: mat stream, dang ket noi lai "
            f"({self.reconnect_count}/{self.max_reconnects})..."
        )
        self.capture = open_capture(
            self.source,
            attempts=self.open_attempts,
            camera_name=self.name,
        )
        if not self.capture.isOpened() and self.reconnect_count >= self.max_reconnects:
            self.exhausted = True
        return False, None

    def close(self) -> None:
        self.capture.release()


class LatestFrameCapture:
    """Continuously drain an RTSP source and expose only its newest frame."""

    def __init__(self, stream: ResilientCapture) -> None:
        self.stream = stream
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame: np.ndarray | None = None
        self._generation = 0
        self._delivered_generation = 0

    @property
    def exhausted(self) -> bool:
        return self.stream.exhausted

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._drain,
            name=f"latest-frame-{self.stream.name}",
            daemon=True,
        )
        self._thread.start()

    def _drain(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.stream.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._generation += 1
                continue
            if self.stream.exhausted:
                break
            time.sleep(0.005)

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if (
                self._frame is None
                or self._generation == self._delivered_generation
            ):
                return False, None
            self._delivered_generation = self._generation
            return True, self._frame

    def close(self) -> None:
        self._stop.set()
        self.stream.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def low_latency_capture(
    stream: ResilientCapture,
) -> ResilientCapture | LatestFrameCapture:
    """Use a latest-frame reader for RTSP without accelerating local files."""
    source = normalize_source(stream.source)
    if not (isinstance(source, str) and source.lower().startswith("rtsp://")):
        return stream
    latest = LatestFrameCapture(stream)
    latest.start()
    return latest


def execute_voice_command(
    cmd: dict,
    *,
    greeter=None,
    find_person=None,
    save_snapshot=None,
    day: str = "",
    now_s: float = 0.0,
) -> tuple[bool, str]:
    """Thuc thi lenh tu voice (chao/chup anh), tra (ok, message de noi lai)."""
    action = str(cmd.get("action") or "").strip().lower()
    if action == "greet":
        name = str(cmd.get("name") or "").strip()
        if greeter is None:
            return False, "loa chào chưa sẵn sàng."
        found = find_person(name) if (find_person and name) else None
        if found is None:
            return False, f"Không thấy {name or 'người này'} trong phòng."
        person_id, display_name = found
        try:
            ok = greeter.wave_greet(
                day=day, person_id=person_id, display_name=display_name,
                now_s=now_s, proactive=True)
        except Exception as error:  # noqa: BLE001
            return False, f"chào {display_name} lỗi: {error}"
        return ok, (f"Đã chào {display_name}."
                    if ok else f"{display_name} vừa được chào rồi.")
    if action == "snapshot":
        if save_snapshot is None:
            return False, "camera chưa sẵn sàng chụp."
        try:
            path = save_snapshot(str(cmd.get("camera") or "A"))
        except Exception as error:  # noqa: BLE001
            return False, f"chụp ảnh lỗi: {error}"
        if not path:
            return False, "chụp ảnh thất bại (chưa có khung hình)."
        return True, f"Đã chụp ảnh, lưu tại {path}."
    return False, f"lệnh {action or '?'} chưa hỗ trợ."


def supervise_halinh(
    script: str | Path,
    stop_event: threading.Event,
    max_restarts: int = 3,
    extra_args: list[str] | None = None,
) -> None:
    """Chay voice Ha Linh BAN CAMERA subprocess, tu restart khi crash.

    Dung khi pipeline dung (stop_event) hoac het luot restart. Log voice
    giu tag [HaLinh]/[HaLinh-sup] rieng de phan biet voi log tracking.
    Mic RTSP camera + loa camera P2P (doi lap voi ban _local).
    """
    restarts = 0
    cmd = [sys.executable, "-u", str(script), *(extra_args or [])]
    while not stop_event.is_set():
        print(f"[HaLinh-sup] khoi dong voice ({' '.join(cmd)})...", flush=True)
        try:
            proc = subprocess.Popen(cmd)
        except OSError as error:
            print(f"[HaLinh-sup] khong chay duoc voice: {error}", flush=True)
            return
        while proc.poll() is None:
            if stop_event.wait(timeout=1.0):
                try:
                    proc.terminate()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=10.0)
                except subprocess.SubprocessError:
                    try:
                        proc.kill()
                    except OSError:
                        pass
                print("[HaLinh-sup] voice da dung theo pipeline.", flush=True)
                return
        code = proc.returncode
        if stop_event.is_set():
            return
        restarts += 1
        print(f"[HaLinh-sup] voice thoat (code={code}) lan {restarts}/"
              f"{max(0, max_restarts) + 1}.", flush=True)
        if restarts > max(0, max_restarts):
            print("[HaLinh-sup] het luot restart, tracking van chay.", flush=True)
            return
        time.sleep(2.0)


def imou_url(channel: int, subtype: int = 1) -> str | None:
    ip = os.getenv("IMOU_IP", "")
    user = os.getenv("IMOU_USER", "")
    password = os.getenv("IMOU_PASSWORD", "")
    if not ip or not user or not password:
        return None
    pw = quote(password, safe="")
    return f"rtsp://{user}:{pw}@{ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}"


def source_label(source: int | str) -> str:
    """Describe a source without exposing RTSP credentials in terminal logs."""
    if isinstance(source, int):
        return f"webcam {source}"
    if "://" in source:
        return "RTSP stream"
    return source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracking ID 2 camera + voice Ha Linh camera (YOLO + ByteTrack + Re-ID).")
    parser.add_argument("--config", type=Path, default=_PROJECT_ROOT / "config/default.yaml",
                        help="Duong dan config (mac dinh: config/default.yaml cua repo, khoi can truyen).")
    parser.add_argument("--source-a", default=None, help="Override camera A.")
    parser.add_argument("--source-b", default=None, help="Override camera B.")
    parser.add_argument("--halinh", dest="halinh",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Luon bat voice Ha Linh BAN CAMERA (mic RTSP + loa P2P) "
                             "song song tracking. Mac dinh bat, tat bang --no-halinh.")
    parser.add_argument("--halinh-script", default=None,
                        help="Duong dan voice script (mac dinh scripts/halinh_assistant.py).")
    parser.add_argument("--halinh-restarts", type=int, default=3,
                        help="So lan tu restart voice khi crash (mac dinh 3).")
    parser.add_argument("--halinh-wake-model", default="zipformer",
                        help="Model STT vong cho wake 'Ha Linh oi' (zipformer/small/tiny).")
    parser.add_argument("--halinh-cmd-model", default="zipformer",
                        help="Model STT vong nghe lenh cua voice (zipformer/medium/small).")
    parser.add_argument("--halinh-channel", type=int, default=None,
                        help="Kenh loa P2P cho Ha Linh (mac dinh p2p_channel trong config).")
    parser.add_argument(
        "--model",
        help="Override YOLO weights from config, for example yolo26s.pt.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        help="Override YOLO inference image size from config.",
    )
    parser.add_argument("--channel-a", type=int, default=1, help="Kenh IMOU cho cam A.")
    parser.add_argument("--channel-b", type=int, default=2, help="Kenh IMOU cho cam B.")
    parser.add_argument(
        "--subtype-a", type=int, default=0, choices=(0, 1), help="0=main, 1=sub stream."
    )
    parser.add_argument(
        "--subtype-b", type=int, default=0, choices=(0, 1), help="0=main, 1=sub stream."
    )
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--open-attempts",
        type=int,
        default=3,
        help="So lan thu mo moi RTSP stream (mac dinh: 3).",
    )
    parser.add_argument(
        "--max-reconnects",
        type=int,
        default=3,
        help="So lan ket noi lai neu stream bi mat khi dang chay.",
    )
    parser.add_argument("--min-area", type=float, default=2000.0,
                        help="Bo box nguoi nho hon nguong (px^2) de giam nhieu.")
    parser.add_argument("--away-grace-s", type=float, default=5.0,
                        help="Business: absent/seat-leave over N sec -> AWAY.")
    parser.add_argument("--out-after-s", type=float, default=60.0,
                        help="Business: absent over N sec -> POSSIBLY_OUT.")
    parser.add_argument("--return-stable-s", type=float, default=2.0,
                        help="Business: stable presence N sec -> WORKING.")
    parser.add_argument("--move-ratio", type=float, default=None,
                        help="Interim displacement threshold 0..1 (default from "
                        "workstate.move_ratio, 0 = presence-only).")
    parser.add_argument("--stream-port", type=int, default=8765,
                        help="Live MJPEG port for the dashboard (0 disables).")
    parser.add_argument("--stream-host", default="127.0.0.1",
                        help="Live MJPEG bind host.")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable the live dashboard stream.")
    parser.add_argument("--supervisor-port", type=int, default=8766,
                        help="Embedded supervisor port cho dashboard "
                        "Pipeline card (VITE_CONTROL_URL, 0 disables).")
    parser.add_argument("--supervisor-host", default="127.0.0.1",
                        help="Embedded supervisor bind host.")
    parser.add_argument("--no-supervisor", action="store_true",
                        help="Tat embedded supervisor (dashboard mat nut Start/Stop).")
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu / auto (mac dinh lay theo config).")
    parser.add_argument("--no-face", action="store_true",
                        help="Tat nhan dien khuon mat ca 2 kenh (chi tracking).")
    parser.add_argument("--face-channels", default=None,
                        help="Override face.channels trong config "
                        "(vi du 'B' chi diem danh 1 cam, 'AB' ca 2).")
    parser.add_argument("--greet", dest="greet",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Bat voice ra loa camera: gio ban tay 4-5 ngon -> chao ten "
                        "(quen) / chao khach (la), thong nhat 10s. Mac dinh bat, tat bang --no-greet.")
    parser.add_argument("--no-face-greet", action="store_true",
                        help="Chi gio tay moi chao; tat chao tu dong khi nhan dien mat "
                        "(mac dinh da palm-only).")
    parser.add_argument("--voice-trigger", dest="voice_trigger",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="DEPRECATED (da bo hello imou): giu de tuong thich CLI, "
                        "khong con tac dung.")
    parser.add_argument("--identity-log", type=Path, default=None,
                        help="Write identity predictions CSV for replay evaluation.")
    parser.add_argument("--metrics-log-s", type=float, default=0.0,
                        help="In StageMetrics moi N giay de soi bottleneck "
                        "(mac dinh 0 = tat).")
    return parser.parse_args()


def normalize_frame_size(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize moi source ve kich thuoc chung de hien thi on dinh."""
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    interpolation = cv2.INTER_AREA if frame.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def _track_floor_points(tracks, projector) -> dict[int, tuple[float, float]]:
    """Floor-meter foot points (homography) for workstation ROI logic."""
    pos: dict[int, tuple[float, float]] = {}
    for track in tracks:
        try:
            pos[track.track_id] = projector.project(track.bbox.foot_point)
        except (ArithmeticError, ValueError) as error:
            print(f"Projection skipped for G{track.track_id}: {error}")
            continue
    return pos


def _track_norm_centers(tracks, width: int, height: int) -> dict[int, tuple[float, float]]:
    """Normalized 0..1 bbox centers for interim displacement mode."""
    pos: dict[int, tuple[float, float]] = {}
    for track in tracks:
        pos[track.track_id] = (
            (track.bbox.x1 + track.bbox.x2) / 2 / max(1, width),
            (track.bbox.y1 + track.bbox.y2) / 2 / max(1, height),
        )
    return pos


def _sane_box(bbox, frame_width: int, frame_height: int) -> bool:
    """Reject absurd boxes (reflections, merged blobs) before tracking."""
    w = max(1.0, bbox.width)
    h = max(1.0, bbox.height)
    if w / h > 3.0 or h / w > 4.0:
        return False
    if bbox.width < 8 or bbox.height < 16:
        return False
    return not (
        bbox.x2 < 0
        or bbox.y2 < 0
        or bbox.x1 > frame_width
        or bbox.y1 > frame_height
    )


def draw_business_states(frame: np.ndarray, tracks, states: dict[int, str]) -> np.ndarray:
    """Ve them business state ke ben box cua tung Global ID (chi de hien thi)."""
    for track in tracks:
        label = f"G{track.track_id}:{states.get(track.track_id, '?')}"
        x1, y1 = max(0, int(track.bbox.x1)), max(0, int(track.bbox.y1))
        cv2.putText(frame, label, (x1, max(20, y1 - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def _person_crop(frame: np.ndarray, bbox) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1 = max(0, min(w, round(bbox.x1)))
    y1 = max(0, min(h, round(bbox.y1)))
    x2 = max(0, min(w, round(bbox.x2)))
    y2 = max(0, min(h, round(bbox.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _flush_queue(
    queue,
    supabase,
    *,
    defer_attendance: bool = False,
    attendance_only: bool = False,
) -> tuple[int, int]:
    """Day pending writes len Supabase. Tra (ok, fail). Gap loi thi dung lai."""
    ok = fail = 0
    for row_id, kind, payload, _attempts in queue.peek(100):
        is_check_in = kind == "event" and payload.get("event") == "CHECK_IN"
        is_known_crop = kind == "face_crop" and "/known/" in payload.get(
            "storage_path", ""
        ).replace("\\", "/")
        is_attendance = kind == "attendance" or is_check_in or is_known_crop
        if attendance_only and not is_attendance:
            continue
        if defer_attendance and is_attendance:
            continue
        done = False
        if kind == "person":
            done = supabase.upsert_person(payload)
        elif kind == "attendance":
            done = supabase.upsert_attendance(payload)
        elif kind == "room_status":
            done = supabase.upsert_room_status(payload)
        elif kind == "current_state":
            done = supabase.upsert_current_state(payload)
        elif kind == "event":
            done = supabase.insert_event(payload)
        elif kind == "face_crop":
            done = supabase.upload_face_crop(
                payload.get("local_path", ""), payload.get("storage_path", "")
            )
        if done:
            queue.ack(row_id)
            ok += 1
        else:
            queue.bump(row_id)
            fail += 1
            # One stale/unsupported row (for example an old schema write)
            # must not block newer attendance records behind it.
            continue
    return ok, fail


def _pending_attendance_rows(
    attendance,
    queue,
    sent_keys: set[tuple[str, str]],
) -> list[dict]:
    """Serialize in-memory and queued attendance for the dashboard."""
    rows = {
        (record.day, record.person_id): {
            "date": record.day,
            "person_id": record.person_id,
            "person_name": record.display_name,
            "global_id": record.global_id,
            "check_in_at": record.wall_time,
            "face_score": record.face_score,
        }
        for record in attendance.records_for_day(
            datetime.now().astimezone().date().isoformat()
        )
    }
    for _row_id, kind, payload, _attempts in queue.peek(500):
        if kind == "attendance":
            rows[(payload["date"], payload["person_id"])] = payload
    return [
        row for key, row in rows.items()
        if key not in sent_keys
    ]


def _store_or_queue(queue, supabase, kind: str, payload: dict) -> None:
    """Queue a write; the inference loop must never block on Supabase."""
    queue.push(kind, payload)


def _merge_same_person_stale(
    day_cache, write_queue, supabase, gid_alias,
    gid_to_person, gid_to_display,
    day_str: str, canon_gid: int, person_id: str, display_name: str,
    manager, business_trackers=(), now_s: float | None = None,
    channel: str = "",
) -> None:
    """Alias duplicate Global IDs of the same face-matched person.

    Fixes the G1-OUT + G2-Working split-brain: when the same human was
    fragmented across IDs, the older Global ID is kept as canonical and the
    newer duplicate is backfilled with person info + merged_into.

    Face-confirmed duplicates are merged even in overlapping camera views.
    """
    matching_gids = {
        gid for gid, pid in gid_to_person.items()
        if pid == person_id and gid not in gid_alias
    }
    # Include identities that were previously face-bound but are currently
    # Away/TEMP_LOST and therefore no longer present in the live maps.
    matching_gids.update(
        gid for gid, record in manager.identities.items()
        if record.employee_id == person_id and gid not in gid_alias
    )
    candidate_gids = {canon_gid, *matching_gids}
    verified_gids = {
        gid for gid in candidate_gids
        if manager.employee_id_of(gid) == person_id
        or gid_to_person.get(gid) == person_id
    }
    canonical_gid = min(verified_gids or candidate_gids)
    for old_gid in sorted(matching_gids):
        if old_gid == canonical_gid:
            continue
        if not manager.merge_identity(old_gid, canonical_gid):
            continue
        gid_alias[old_gid] = canonical_gid
        gid_to_person[canonical_gid] = person_id
        gid_to_display[canonical_gid] = display_name
        gid_to_person[old_gid] = person_id
        gid_to_display[old_gid] = display_name
        for tracker in business_trackers:
            try:
                tracker.transfer_assignment(old_gid, canonical_gid)
            except (KeyError, ValueError) as error:
                print(f"Workstate transfer G{old_gid}->{canonical_gid} skipped: {error}")
        cached = day_cache.status_of(day_str, old_gid)
        _store_or_queue(write_queue, supabase, "room_status", {
            "date": day_str, "global_id": old_gid,
            "person_id": person_id, "person_name": display_name,
            "in_room": cached.in_room if cached else True,
            "label": cached.label if cached else LABEL_UNKNOWN,
            "merged_into": canonical_gid,
        })
        print(f"[Reconcile] G{old_gid} -> G{canonical_gid} {display_name}")


def _candidate_vote_reached(
    streaks: dict[int, list],
    gid: int,
    employee_id: str | None,
    score: float,
    now_s: float,
    *,
    required_hits: int = 3,
    window_s: float = 10.0,
    min_score: float = 0.45,
) -> str | None:
    """Dem streak face-match chua-consensus cho 1 GID (pure, de test).

    Moi observation ``match.is_known`` +1 cho employee do. Quan sat
    unknown (khong match / score thap) giu streak cu (chiu flicker khi
    goc mat xau xen ke), employee known khac reset ve 0. Du
    ``required_hits`` trong ``window_s`` thi tra employee_id (caller
    merge) va reset streak de khong merge lap; het window thi het han.
    """
    if not employee_id or score < min_score:
        return None
    cur = streaks.get(gid)
    if cur is None or cur[0] != employee_id or (now_s - cur[2]) > window_s:
        cur = [employee_id, 0, now_s, now_s]
        streaks[gid] = cur
    cur[1] += 1
    cur[3] = now_s
    if cur[1] >= max(2, int(required_hits)):
        streaks.pop(gid, None)
        return employee_id
    return None


def _merge_unknown_gid_by_face(
    manager, gid_alias, gid_to_person, gid_to_display,
    gid_to_face_candidates, day_cache, write_queue, supabase,
    business_trackers, day_str, raw_gid, employee_id, display_name,
) -> bool:
    """Gop GID unknown (chua bind) vao GID da bind cung employee.

    Chi dung cho face-provisional (chua consensus): ReID cross-cam tach
    GID moi, face candidates lap lai cung employee da bind o GID khac.
    Khong tick diem danh / grow prototype / save crop (giu cho consensus).
    Nguong ReID giu nguyen.
    """
    if raw_gid in gid_alias:
        return False
    if raw_gid not in manager.identities:
        return False
    if manager.employee_id_of(raw_gid) is not None:
        return False
    if gid_to_person.get(raw_gid) is not None:
        return False
    conflict_gid = next(
        (gid for gid, record in manager.identities.items()
         if gid != raw_gid
         and gid not in gid_alias
         and record.employee_id == employee_id),
        None,
    )
    if conflict_gid is None:
        return False
    if not manager.merge_identity(raw_gid, conflict_gid):
        print(f"[Identity conflict] {employee_id}: keep G{conflict_gid}; "
              f"G{raw_gid} stays UNKNOWN (face-provisional)")
        return False
    gid_alias[raw_gid] = conflict_gid
    if raw_gid in gid_to_face_candidates:
        gid_to_face_candidates[conflict_gid] = gid_to_face_candidates[raw_gid]
    gid_to_person[conflict_gid] = employee_id
    gid_to_display[conflict_gid] = display_name
    gid_to_person[raw_gid] = employee_id
    gid_to_display[raw_gid] = display_name
    for tracker in business_trackers or ():
        try:
            tracker.transfer_assignment(raw_gid, conflict_gid)
        except (KeyError, ValueError) as error:
            print(f"Workstate transfer G{raw_gid}->{conflict_gid} skipped: {error}")
    cached = day_cache.status_of(day_str, raw_gid)
    _store_or_queue(write_queue, supabase, "room_status", {
        "date": day_str, "global_id": raw_gid,
        "person_id": employee_id, "person_name": display_name,
        "in_room": cached.in_room if cached else True,
        "label": cached.label if cached else LABEL_UNKNOWN,
        "merged_into": conflict_gid,
    })
    print(f"[Reconcile] G{raw_gid} -> G{conflict_gid} "
          f"{display_name} (face-provisional)")
    return True


def _remap_tracks(
    tracks: list[Track],
    gid_alias: dict[int, int],
    manager,
) -> list[Track]:
    """Replace live duplicate Global IDs after an identity merge."""
    remapped: list[Track] = []
    for track in tracks:
        gid = track.track_id
        seen: set[int] = set()
        while gid in gid_alias and gid not in seen:
            seen.add(gid)
            gid = gid_alias[gid]
        if gid == track.track_id:
            remapped.append(track)
            continue
        remapped.append(
            Track(
                track_id=gid,
                bbox=track.bbox,
                confidence=track.confidence,
                age=track.age,
                hits=track.hits,
                confirmed=track.confirmed,
                local_track_id=track.local_track_id,
                global_person_id=gid,
                employee_id=manager.employee_id_of(gid),
            )
        )
    return remapped


def _reconcile_unknowns(
    day_cache, write_queue, supabase, reconciler, gid_alias,
    gid_to_person, gid_to_display, unknown_of_gid,
    day_str: str, canon_gid: int, person_id: str, display_name: str,
    person_embedding, business_trackers=(),
) -> None:
    """Merge stale unknown GIDs verified as this person into canon_gid.

    Updates BOTH sides: the old unknown row gets person info + merged_into,
    the canonical gid keeps receiving live writes. Workstation assignment
    moves to the canonical gid. Dashboard hides merged rows, so the person
    never appears twice.
    """
    merges = reconciler.find_merges(person_embedding, unknown_of_gid)
    for old_gid, owner in merges.items():
        if old_gid == canon_gid or old_gid in gid_alias:
            continue
        gid_alias[old_gid] = canon_gid
        gid_to_person[old_gid] = person_id
        gid_to_display[old_gid] = display_name
        for tracker in business_trackers:
            try:
                tracker.transfer_assignment(old_gid, canon_gid)
            except (KeyError, ValueError) as error:
                print(f"Workstate transfer G{old_gid}->{canon_gid} skipped: {error}")
        cached = day_cache.status_of(day_str, old_gid)
        _store_or_queue(write_queue, supabase, "room_status", {
            "date": day_str, "global_id": old_gid,
            "person_id": person_id, "person_name": display_name,
            "in_room": cached.in_room if cached else True,
            "label": cached.label if cached else LABEL_UNKNOWN,
            "merged_into": canon_gid,
        })
        supabase.reassign_face_crops(day_str, owner, person_id)
        reconciler.forget(owner)
        print(f"[Reconcile] G{old_gid} ({owner}) -> G{canon_gid} {display_name}")


def _prune_identity_maps(
    manager, gid_to_person, gid_to_display, gid_to_score,
    unknown_of_gid, reconciler, fusion, business_trackers=(),
    face_consumer=None, attendance=None, now_s: float = 0.0,
) -> None:
    """Drop per-ID memory for identities the manager already retired.

    Dead ghost IDs must not linger in overlay memory, seat assignment or
    the dashboard live list. DB history rows are kept (audit).
    """
    alive = set(manager.identities)
    if face_consumer is not None:
        face_consumer.forget_retired(alive)
    for store in (gid_to_person, gid_to_display, gid_to_score):
        for gid in [g for g in store if g not in alive]:
            store.pop(gid, None)
    for gid in [g for g in unknown_of_gid if g not in alive]:
        reconciler.forget(unknown_of_gid.pop(gid))
    for tracker in business_trackers:
        tracker.forget_retired(alive)
    for gid in [g for g in list(fusion._was_out) if g not in alive]:
        fusion.forget(gid)
    if attendance is not None:
        try:
            attendance.prune(now_s, alive)
        except (AttributeError, TypeError, ValueError):
            pass


def _snapshot_new_gids(
    seen_gids: set[int],
    last_tracks_a, last_tracks_b,
    frame_a, frame_b, ghosts_dir: Path, cap: int = 100,
) -> None:
    """Save the first crop of every new Global ID for ghost inspection."""
    if len(seen_gids) >= cap:
        return
    try:
        ghosts_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    owners: dict[int, tuple] = {}
    if frame_a is not None:
        for track in last_tracks_a:
            owners.setdefault(track.track_id, (frame_a, track.bbox, "A"))
    if frame_b is not None:
        for track in last_tracks_b:
            owners.setdefault(track.track_id, (frame_b, track.bbox, "B"))
    for gid, (frame, bbox, channel) in owners.items():
        if gid in seen_gids or len(seen_gids) >= cap:
            continue
        seen_gids.add(gid)
        try:
            crop = _person_crop(frame, bbox)
            if crop is not None and crop.size > 0:
                cv2.imwrite(str(ghosts_dir / f"G{gid}_{channel}_first.jpg"), crop)
        except (OSError, ValueError, cv2.error) as error:
            print(f"Ghost snapshot G{gid} skipped: {error}")


def main() -> None:
    # Dìm log decode ffmpeg của OpenCV (hevc "Could not find ref" / "Error
    # constructing the frame RPS" do substream camera, frame vẫn chảy bình
    # thường). Đặt env của user thắng nếu đã export trước.
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "fatal")
    args = parse_args()
    config = load_config(args.config)

    source_a = normalize_source(
        args.source_a
        or imou_url(args.channel_a, args.subtype_a)
        or config.camera.source
    )
    source_b = normalize_source(
        args.source_b or imou_url(args.channel_b, args.subtype_b)
    )
    if source_b is None:
        print("Thieu --source-b va khong co IMOU_* trong .env. "
              "Vi du: --source-b data/samples/hallway.mp4")
        raise SystemExit(2)
    device = resolve_device(args.device or config.detection.device)
    model_path = args.model or config.detection.model_path
    image_size = args.imgsz or config.detection.image_size
    # Thuần GPU: YOLO phải chạy cuda/TensorRT, không fallback CPU lặng lẽ.
    if device != "cuda":
        print(f"CẢNH BÁO pure-GPU: YOLO device={device}, kỳ vọng cuda. "
              f"Kiểm tra torch cuda / --device cuda.")
    # Xác nhận TensorRT engine, fallback .pt (vẫn cuda) nếu thiếu TRT.
    if str(model_path).endswith(".engine") and not Path(model_path).is_file():
        fallback = Path(str(model_path)).with_suffix(".pt")
        if fallback.is_file():
            print(f"Thiếu {model_path}, fallback {fallback} (vẫn cuda, chậm hơn TRT).")
            model_path = str(fallback)
        else:
            print(f"Thiếu {model_path} và {fallback}, YOLO sẽ lỗi khi load.")
    print(
        f"Device: {device} | Model: {model_path} @ {image_size}px | "
        f"Camera A: {source_label(source_a)} | "
        f"Camera B: {source_label(source_b)}"
    )
    if str(model_path).endswith(".engine"):
        print("YOLO: TensorRT engine (FP16, batch A+B) — kỳ vọng ~150+ fps trên GPU.")
    else:
        print("YOLO: .pt cuda (chậm hơn engine ~3x) — export engine để tối ưu.")

    detector = YoloPersonDetector(
        model_path=model_path,
        confidence=config.detection.confidence_threshold,
        person_class_id=config.detection.person_class_id,
        image_size=image_size,
        device=device,
        nms_iou_threshold=config.detection.nms_iou_threshold,
        nested_box_containment_threshold=(
            config.detection.nested_box_containment_threshold
        ),
    )
    effective_fps = max(1, round(config.camera.fps / config.camera.process_every_n_frames))
    tracker_a = ByteTrackTracker(
        frame_rate=effective_fps,
        track_buffer=config.tracking.track_buffer,
        track_high_threshold=config.tracking.track_high_threshold,
        track_low_threshold=config.tracking.track_low_threshold,
        new_track_threshold=config.tracking.new_track_threshold,
        match_threshold=config.tracking.byte_match_threshold,
        min_hits=config.tracking.min_hits,
    )
    tracker_b = ByteTrackTracker(
        frame_rate=effective_fps,
        track_buffer=config.tracking.track_buffer,
        track_high_threshold=config.tracking.track_high_threshold,
        track_low_threshold=config.tracking.track_low_threshold,
        new_track_threshold=config.tracking.new_track_threshold,
        match_threshold=config.tracking.byte_match_threshold,
        min_hits=config.tracking.min_hits,
    )
    identity_cfg = config.identity
    # Face identification is independent from body ReID continuity.
    from camera_tracking.face import InsightFaceEmbedder as _IFEmbedder
    face_cfg_early = config.face
    if args.face_channels is not None:
        face_cfg_early = face_cfg_early.model_copy(
            update={"channels": [c for c in args.face_channels.upper() if c in ("A", "B")] or ["B"]}
        )
    _need_iface = bool(
        face_cfg_early.enabled and not args.no_face and face_cfg_early.channels
    )
    shared_face_embedder = None
    if _need_iface:
        try:
            shared_face_embedder = _IFEmbedder(
                model_pack=face_cfg_early.model_pack,
                det_size=face_cfg_early.det_size,
                device=face_cfg_early.face_device,
            )
            # Validate truoc khi mo camera (tai model nang lan dau o day).
            shared_face_embedder._load()
            print(f"InsightFace {face_cfg_early.model_pack} "
                  f"({shared_face_embedder.resolved_device})")
            if shared_face_embedder.resolved_device != "cuda":
                print("CẢNH BÁO pure-GPU: InsightFace đang CPU. "
                      "Cài onnxruntime-gpu đúng bản CUDA để face chạy GPU: "
                      "pip uninstall onnxruntime onnxruntime-gpu -y && "
                      "pip install onnxruntime-gpu")
        except Exception as error:  # noqa: BLE001 - face may be disabled explicitly
            print(f"InsightFace unavailable; employee recognition disabled: {error}")
            shared_face_embedder = None
    embedder = OsnetEmbedding(
        model_name=identity_cfg.reid_model,
        device=identity_cfg.reid_device,
    )
    try:
        embedder.load()
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "Production tracking requires OSNet; install the reid extra and "
            "make the model available. No histogram fallback is allowed."
        ) from error
    _reid_dev = getattr(embedder, "_device", None)
    print(f"ReID: OSNet {identity_cfg.reid_model} ({_reid_dev or identity_cfg.reid_device}, "
          f"FP16 cuda, event-driven: chỉ new/re-entry/cross-camera, "
          f"refresh={identity_cfg.gallery_refresh_steps})")

    # Mot GlobalIdentityManager dung chung cho ca 2 channel -> Global ID
    # xuyen tracklet, xuyen mat dau dai, xuyen channel.
    manager = GlobalIdentityManager(
        embedder,
        GlobalIdentityConfig(
            gallery_size=identity_cfg.gallery_size,
            appearance_weight=identity_cfg.appearance_weight,
            spatial_weight=identity_cfg.spatial_weight,
            time_weight=identity_cfg.time_weight,
            channel_weight=identity_cfg.channel_weight,
            match_threshold=identity_cfg.match_threshold,
            max_center_distance_ratio=identity_cfg.max_center_distance_ratio,
            same_camera_reconnect_distance_ratio=(
                identity_cfg.same_camera_reconnect_distance_ratio
            ),
            same_camera_reconnect_s=identity_cfg.same_camera_reconnect_s,
            min_appearance_similarity=identity_cfg.min_appearance_similarity,
            named_appearance_floor=identity_cfg.named_appearance_floor,
            named_floor_locked=identity_cfg.named_floor_locked,
            face_anchor_ttl_s=identity_cfg.face_anchor_ttl_s,
            face_anchor_ttl_steps=identity_cfg.face_anchor_ttl_steps,
            occlusion_iou_threshold=identity_cfg.occlusion_iou_threshold,
            occlusion_area_jump_ratio=identity_cfg.occlusion_area_jump_ratio,
            gallery_append_min_sim=identity_cfg.gallery_append_min_sim,
            active_duplicate_similarity=identity_cfg.active_duplicate_similarity,
            tentative_min_hits=identity_cfg.tentative_min_hits,
            temp_lost_s=identity_cfg.temp_lost_s,
            long_lost_s=identity_cfg.long_lost_s,
            unresolved_keep_s=identity_cfg.unresolved_keep_s,
            min_gallery_confidence=identity_cfg.min_gallery_confidence,
            gallery_refresh_steps=identity_cfg.gallery_refresh_steps,
        ),
    )
    identity_store = DailyIdentityStore(
        identity_cfg.state_db,
        model_key=f"osnet:{identity_cfg.reid_model}",
    )
    startup_day = datetime.now().astimezone().date().isoformat()
    current_identity_day = startup_day
    restored_identities = identity_store.load(startup_day, manager)
    if restored_identities:
        print(f"Restored {restored_identities} identities for {startup_day}")
    # Business theo channel: position (workstation ROI) + presence,
    # khong bao gio anh huong ID. Channel A (room) dung ROI; channel B
    # (door) presence-only. Trong workstations = fallback presence-only.
    from camera_tracking.analytics import FloorProjector
    try:
        projector = FloorProjector(
            config.analytics.calibration.image_points,
            config.analytics.calibration.floor_points,
        )
    except Exception as error:  # noqa: BLE001 - bad calibration points
        print(f"Homography loi ({error}), workstation ROI tat.")
        projector = None
    ws_zones = [
        WorkstationZone(name=ws.name,
                        core=[(float(x), float(y)) for x, y in ws.core],
                        extended=[(float(x), float(y)) for x, y in ws.extended])
        for ws in config.workstations
    ]
    ws_cfg = config.workstate
    move_ratio = args.move_ratio if args.move_ratio is not None else ws_cfg.move_ratio
    use_roi = bool(ws_zones) and projector is not None
    if use_roi:
        print(f"Workstations: {', '.join(z.name for z in ws_zones)} "
              f"(floor meters, homography)")
    elif move_ratio > 0:
        print(f"Workstations: none -> interim displacement mode "
              f"(move_ratio={move_ratio}).")
    else:
        print("Workstations: none configured -> presence-only business logic.")
    business_a = ChannelBusinessTracker(
        channel="A",
        away_grace_s=args.away_grace_s,
        out_after_s=args.out_after_s,
        return_stable_s=args.return_stable_s,
        workstations=ws_zones if use_roi else [],
        grace_s=ws_cfg.grace_s,
        dwell_s=ws_cfg.dwell_s,
        assign_dwell_s=ws_cfg.assign_dwell_s,
        hysteresis_m=ws_cfg.hysteresis_m,
        motion_influence=ws_cfg.motion_influence,
        person_map=dict(ws_cfg.person_map),
        prune_after_s=ws_cfg.prune_after_s,
        move_ratio=move_ratio,
        settle_ratio=ws_cfg.settle_ratio,
    )
    business_b = ChannelBusinessTracker(
        channel="B",
        away_grace_s=args.away_grace_s,
        out_after_s=args.out_after_s,
        return_stable_s=args.return_stable_s,
        # Channel B là camera cửa (transit): luôn presence-only, không dùng
        # workstation ROI của phòng A để khỏi gán nhầm Working/Near seat.
        workstations=[],
        grace_s=ws_cfg.grace_s,
        dwell_s=ws_cfg.dwell_s,
        assign_dwell_s=ws_cfg.assign_dwell_s,
        hysteresis_m=ws_cfg.hysteresis_m,
        motion_influence=ws_cfg.motion_influence,
        person_map={},
        prune_after_s=ws_cfg.prune_after_s,
        move_ratio=0.0,
        settle_ratio=ws_cfg.settle_ratio,
    )
    workstate_a = WorkstateConsumer(business_a)
    workstate_b = WorkstateConsumer(business_b)
    count_a = StablePersonCount(rise_frames=3, fall_frames=12)
    count_b = StablePersonCount(rise_frames=3, fall_frames=12)

    # --- Face diem danh + daily store + room fusion ---
    # Tat ca optional: thieu model/lib/.env van chay tracking nhu cu.
    # Dung lai shared_face_embedder da load o tren (ReID mat) neu co.
    face_cfg = face_cfg_early
    face_threshold = face_cfg.match_threshold
    use_face = bool(face_cfg.enabled and not args.no_face and face_cfg.channels)
    face_embedder = shared_face_embedder
    face_matcher = face_gallery = attendance = face_consumer = None
    face_model_lock = threading.Lock()
    enrollment_lock = threading.Lock()
    gid_to_person: dict[int, str] = {}
    gid_to_display: dict[int, str] = {}
    gid_to_score: dict[int, float] = {}
    gid_to_face_candidates: dict[int, list[tuple[str, float]]] = {}
    # Full candidate objects are kept separately from the serializable UI
    # labels so identities can be assigned jointly across all visible GIDs.
    gid_to_face_rankings: dict[int, list[tuple[object, float]]] = {}
    gid_face_ranking_at: dict[int, float] = {}
    face_assignment_streaks: dict[int, list] = {}
    # Most recent recognition quality, used only to schedule background
    # attendance. It never changes the foreground shown on screen.
    face_quality_by_track: dict[tuple[str, int], tuple[float, float]] = {}
    # Streak face-provisional: gid -> [employee_id, hits, first_s, last_s].
    # Gop GID unknown lap lai cung employee da bind (fix split-brain
    # cross-cam) ma khong can doi consensus, khong ha nguong ReID.
    face_candidate_streaks: dict[int, list] = {}
    # (channel, gid) -> (frame time, normalized face bbox in person crop).
    # Any recent detected face is accepted; looking straight is not required.
    recent_face_by_track: dict[
        tuple[str, int], tuple[float, tuple[float, float, float, float]]
    ] = {}
    # Hen chao khach la: gid -> (han phat, day, kenh, at_iso, loa).
    # Cho dinh danh unknown_greet_delay_s truoc khi chao "quy khach";
    # mat xac nhan kip thi nhanh known huy hen (chi chao ten).
    pending_unknown_greet: dict[int, tuple[float, str, str, str, int]] = {}
    unknown_of_gid: dict[int, str] = {}
    unknown_counter = 0
    if use_face and face_embedder is None:
        print("Face disabled (thieu InsightFace embedder).")
        use_face = False
    if use_face:
        try:
            from camera_tracking.face import (
                EnrolledPerson,
                FaceAttendanceService,
                FaceMatcher,
                MatchResult,
                load_gallery,
                unique_face_assignments,
            )
            from camera_tracking.face.gallery import load_registry, save_prototype
            face_gallery = load_gallery(
                face_cfg.gallery_dir,
                face_embedder,
                face_cfg.name_map,
                face_cfg.employee_map,
            )
            face_matcher = FaceMatcher(
                face_gallery,
                threshold=face_threshold,
                min_margin=face_cfg.min_margin,
            )
            from camera_tracking.face import FaceTrackConsumer
            _base_known_cd = float(getattr(face_cfg, "known_cooldown_s", 30.0))
            face_consumer = FaceTrackConsumer(
                face_embedder,
                face_matcher,
                min_person_area_px=face_cfg.min_person_area_px,
                min_face_px=face_cfg.min_face_px,
                min_face_score=face_cfg.min_face_score,
                min_blur_variance=face_cfg.min_blur_variance,
                # Until the frame-level unique-name resolver confirms the
                # winner, a local Top-1 must be observed again promptly. Once
                # bound, FaceWorker applies the normal long recheck cadence.
                known_cooldown_s=min(2.0, _base_known_cd),
                unknown_cooldown_s=float(getattr(
                    face_cfg, "unknown_cooldown_s", 0.6)),
                channels=tuple(face_cfg.channels),
                consensus_hits=face_cfg.consensus_hits,
                consensus_window_s=face_cfg.consensus_window_s,
            )
            enrolled_by_employee = {
                (person.employee_id or person.person_id): person
                for person in face_gallery.people
            }
            for restored_gid, restored in manager.identities.items():
                if not restored.employee_id:
                    continue
                person = enrolled_by_employee.get(restored.employee_id)
                if person is None:
                    continue
                gid_to_person[restored_gid] = restored.employee_id
                gid_to_display[restored_gid] = person.display_name
            attendance = FaceAttendanceService(
                debounce_hits=config.attendance.debounce_hits,
                window_s=config.attendance.window_s,
                active_hour_start=config.attendance.active_hour_start,
                active_hour_end=config.attendance.active_hour_end,
            )
            print(f"Face gallery: {len(face_gallery)} nguoi tu {face_cfg.gallery_dir} "
                  f"| threshold={face_threshold} margin={face_cfg.min_margin} "
                  f"consensus={face_cfg.consensus_hits} "
                  f"| channels={list(face_cfg.channels)}")
        except ImportError as error:
            print(f"Face disabled (thieu module: {error})")
            use_face = False
    # Face async worker: main loop không bao giờ chờ InsightFace.
    # YOLO+ByteTrack chạy realtime, Face chỉ chạy event-driven once-per-track.
    face_worker = None
    if use_face and face_consumer is not None:
        try:
            from camera_tracking.face.worker import FaceWorker as _FaceWorker
            face_worker = _FaceWorker(
                face_consumer,
                max_queue=int(getattr(config.voice, "face_max_queue", 8)),
                model_lock=face_model_lock,
                max_job_age_s=float(getattr(
                    config.voice, "face_max_job_age_s", 5.0
                )),
            )
            face_worker.start()
            print(f"Face worker: async, once-per-track, "
                  f"recheck={min(float(getattr(config.voice, 'face_recheck_s', 30.0)), float(getattr(face_cfg, 'known_cooldown_s', 30.0))):.1f}s, "
                  f"unknown={float(getattr(face_consumer, 'unknown_cooldown_s', 0.6)):.1f}s"
                  f" (x2 khi queue day, tuy may), "
                  f"queue={face_worker.max_queue} (main loop không chờ Face).")
        except ImportError as error:
            print(f"Face worker disabled ({error}), fallback sync (lag).")
            face_worker = None

    from camera_tracking.store.daily import DailyStateCache
    from camera_tracking.store.faces import FaceCropSaver
    from camera_tracking.store.queue import WriteQueue, WriteQueueWorker
    from camera_tracking.store.supabase_client import SupabaseSettings, SupabaseStore

    day_cache = DailyStateCache(
        inroom_min_interval_s=config.room_fusion.inroom_min_interval_s
    )
    crop_saver = FaceCropSaver(
        root=config.store.local_faces_dir,
        max_per_owner_day=config.store.max_crops_per_owner_day,
        jpeg_quality=config.store.face_jpeg_quality,
        max_side_px=config.store.face_max_side_px,
    )
    write_queue = WriteQueue(config.store.queue_db)
    supabase = SupabaseStore(
        SupabaseSettings.from_env(config.store.supabase_enabled)
    )
    write_worker = WriteQueueWorker(
        write_queue,
        lambda: _flush_queue(
            write_queue, supabase, defer_attendance=True
        ),
        interval_s=1.0,
    )
    write_worker.start()
    if config.store.supabase_enabled:
        print("Supabase: asynchronous writer enabled.")
    else:
        print("Supabase: disabled; writes remain in local queue.")
    fusion = RoomPresenceAggregator(
        leave_confirm_window_s=config.room_fusion.leave_confirm_window_s,
        absent_fallback_s=config.room_fusion.absent_fallback_s,
    )
    # Preload attendance đã tick trong ngày từ Supabase để restart giữa ngày
    # không tick trùng (RAM-only trước đây là lỗ hổng thực tế).
    if attendance is not None:
        try:
            from camera_tracking.face.attendance import AttendanceRecord as _AR

            preloaded = []
            for row in supabase.fetch_attendance_day(startup_day):
                try:
                    preloaded.append(_AR(
                        day=str(row.get("date", startup_day)),
                        person_id=str(row.get("person_id", "")),
                        display_name=str(row.get("person_name")
                                         or row.get("person_id", "")),
                        global_id=int(row.get("global_id") or 0),
                        first_seen_at=0.0,
                        wall_time=str(row.get("check_in_at") or ""),
                        face_score=float(row.get("face_score") or 0.0),
                    ))
                except (TypeError, ValueError):
                    continue
            preloaded = [r for r in preloaded if r.person_id]
            if preloaded:
                attendance.preload(preloaded)
                day_cache.preload_attendance(
                    startup_day, [r.person_id for r in preloaded])
                print(f"Preloaded {len(preloaded)} attendance từ DB ({startup_day})")
        except Exception as error:  # noqa: BLE001 - offline vẫn chạy RAM-only
            print(f"Preload attendance bỏ qua (offline): {error}")
    room_status_now: dict = {}
    last_flush_s = 0.0
    last_metrics_log_s = 0.0
    last_runtime_write_s = 0.0
    runtime_prev = {"t": 0.0, "frame": 0}
    # Unknown GID -> canonical (identified) GID after face reconcile.
    gid_alias: dict[int, int] = {}
    reconciler = IdentityReconciler(threshold=face_threshold)

    # --- Live MJPEG stream for the dashboard (2 cameras + model overlay) ---
    stream_on = False
    streamer = None
    jpeg_renderer = None
    if not args.no_stream and args.stream_port:
        from camera_tracking.streaming import LatestJpegRenderer, MjpegStreamer
        streamer = MjpegStreamer(host=args.stream_host, port=args.stream_port)
        stream_on = streamer.start()
        sent_attendance_keys: set[tuple[str, str]] = set()
        attendance_send_lock = threading.Lock()

        def pending_attendance() -> list[dict]:
            if attendance is None:
                return []
            return _pending_attendance_rows(
                attendance, write_queue, sent_attendance_keys
            )

        def send_attendance() -> dict:
            if attendance is None:
                return {"sent": 0, "failed": 0}
            with attendance_send_lock:
                ok, fail = _flush_queue(
                    write_queue, supabase, attendance_only=True
                )
                pending_keys = {
                    (row["date"], row["person_id"])
                    for row_id, kind, row, attempts in write_queue.peek(500)
                    if kind == "attendance"
                }
                for record in attendance.records_for_day(
                    datetime.now().astimezone().date().isoformat()
                ):
                    key = (record.day, record.person_id)
                    if key not in pending_keys:
                        sent_attendance_keys.add(key)
                return {"sent": ok, "failed": fail}

        def register_person(payload: dict) -> dict:
            if (
                not use_face
                or face_embedder is None
                or face_gallery is None
                or face_matcher is None
            ):
                return {
                    "ok": False,
                    "status": 503,
                    "message": "Face recognition chưa sẵn sàng.",
                }
            employee_id = str(payload.get("employee_id", "")).strip()
            display_name = " ".join(str(payload.get("display_name", "")).split())
            image_data = payload.get("image")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", employee_id):
                return {
                    "ok": False,
                    "message": "Mã nhân viên chỉ gồm chữ, số, dấu chấm, gạch ngang hoặc gạch dưới.",
                }
            if not 2 <= len(display_name) <= 100:
                return {"ok": False, "message": "Tên nhân viên không hợp lệ."}
            if payload.get("consent") is not True:
                return {
                    "ok": False,
                    "message": "Cần xác nhận đồng ý xử lý dữ liệu khuôn mặt.",
                }
            if not isinstance(image_data, str):
                return {"ok": False, "message": "Chưa có ảnh đăng ký."}
            encoded = image_data.split(",", 1)[-1]
            try:
                raw_image = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                return {"ok": False, "message": "Ảnh đăng ký không hợp lệ."}
            if not raw_image or len(raw_image) > 5 * 1024 * 1024:
                return {"ok": False, "message": "Ảnh phải nhỏ hơn 5 MB."}
            image_array = np.frombuffer(raw_image, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                return {"ok": False, "message": "Không đọc được định dạng ảnh."}
            height, width = image.shape[:2]
            if max(height, width) > 1920:
                scale = 1920 / max(height, width)
                image = cv2.resize(
                    image,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            with enrollment_lock:
                if any(
                    person.employee_id == employee_id
                    or person.person_id == employee_id
                    for person in face_gallery.people
                ):
                    return {
                        "ok": False,
                        "status": 409,
                        "message": "Mã nhân viên đã được đăng ký.",
                    }
                try:
                    with face_model_lock:
                        detections = face_embedder.detect_embed(image)
                except RuntimeError as error:
                    return {"ok": False, "status": 503, "message": str(error)}
                if len(detections) == 0:
                    return {
                        "ok": False,
                        "message": "Không tìm thấy khuôn mặt rõ ràng trong ảnh.",
                    }
                if len(detections) > 1:
                    return {
                        "ok": False,
                        "message": "Ảnh có nhiều khuôn mặt; hãy dùng ảnh chỉ có một người.",
                    }

                gallery_root = Path(face_cfg.gallery_dir)
                gallery_root.mkdir(parents=True, exist_ok=True)
                image_path = gallery_root / f"{employee_id}.jpg"
                ok, jpeg = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92]
                )
                if not ok:
                    return {"ok": False, "message": "Không thể lưu ảnh đăng ký."}
                temp_image = gallery_root / f".{employee_id}.tmp.jpg"
                temp_image.write_bytes(jpeg.tobytes())
                temp_image.replace(image_path)

                registry = load_registry(gallery_root)
                registry[employee_id] = {
                    "display_name": display_name,
                    "employee_id": employee_id,
                }
                registry_path = gallery_root / "registry.json"
                temp_registry = gallery_root / ".registry.tmp.json"
                temp_registry.write_text(
                    json.dumps(registry, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temp_registry.replace(registry_path)

                detection = detections[0]
                face_gallery.people.append(EnrolledPerson(
                    person_id=employee_id,
                    display_name=display_name,
                    embedding=np.asarray(detection.embedding, dtype=np.float32),
                    source_path=str(image_path),
                    employee_id=employee_id,
                ))
                _store_or_queue(write_queue, supabase, "person", {
                    "person_id": employee_id,
                    "display_name": display_name,
                    "photo_url": str(image_path),
                    "active": True,
                })
            return {
                "ok": True,
                "person_id": employee_id,
                "display_name": display_name,
                "message": "Đăng ký nhân viên thành công.",
            }

        streamer.set_attendance_actions(pending_attendance, send_attendance)
        streamer.set_enrollment_action(register_person)
        if stream_on:
            jpeg_renderer = LatestJpegRenderer(streamer)
            jpeg_renderer.start()
            print(f"Live stream: http://{args.stream_host}:{args.stream_port} "
                  f"(dashboard Live tab, VITE_STREAM_URL)")
        else:
            print(f"Stream port {args.stream_port} busy, "
                  f"running without live stream.")

    # --- Embedded supervisor cho dashboard (VITE_CONTROL_URL, :8766) ---
    # Gộp backend + supervisor vào 1 lệnh: chạy pipeline là dashboard có
    # ngay API status (Pipeline card). Tắt bằng --no-supervisor hoặc
    # --supervisor-port 0. Chạy thread nền, không block inference.
    supervisor_server = None
    _sup_port = 0 if args.no_supervisor else args.supervisor_port
    if _sup_port:
        try:
            sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
            from pipeline_supervisor import serve as _serve_supervisor
            _sup_args = ["--stream-port", str(args.stream_port)]
            if args.no_stream:
                _sup_args = ["--no-stream"]
            if args.greet:
                _sup_args.append("--greet")
            else:
                _sup_args.append("--no-greet")
            if args.voice_trigger:
                _sup_args.append("--voice-trigger")
            else:
                _sup_args.append("--no-voice-trigger")
            if args.no_face:
                _sup_args.append("--no-face")
            supervisor_server = _serve_supervisor(
                args.supervisor_host, _sup_port,
                managed_pid=os.getpid(),
                managed_started_at=time.time(),
                managed_stream_port=args.stream_port if not args.no_stream else None,
                managed_args=_sup_args,
            )
            _sup_thread = threading.Thread(
                target=supervisor_server.serve_forever,
                kwargs={"poll_interval": 0.2},
                name="embedded-supervisor",
                daemon=True,
            )
            _sup_thread.start()
            print(f"Supervisor (embedded): http://{args.supervisor_host}:{_sup_port} "
                  f"(dashboard Pipeline card, VITE_CONTROL_URL)")
        except OSError as error:
            supervisor_server = None
            print(f"Supervisor port {_sup_port} busy ({error}), "
                  f"dashboard mat nut Start/Stop.")
        except (ImportError, ValueError) as error:
            supervisor_server = None
            print(f"Supervisor embedded disabled ({error}).")

    cap_a = open_capture(source_a, attempts=args.open_attempts, camera_name="A")
    opened_a = cap_a.isOpened()
    if not opened_a:
        print(f"Không mở được camera A: {source_label(source_a)}")
    stream_a = low_latency_capture(ResilientCapture(
        source_a,
        "A",
        cap_a,
        args.open_attempts,
        max_reconnects=max(0, args.max_reconnects),
    ))

    # Drain A while B is opening; otherwise A can accumulate the entire B
    # connection time in FFmpeg's internal RTSP buffer.
    cap_b = open_capture(source_b, attempts=args.open_attempts, camera_name="B")
    opened_b = cap_b.isOpened()
    if not opened_b:
        print(f"Không mở được camera B: {source_label(source_b)}")
    if not opened_a and not opened_b:
        stream_a.close()
        raise SystemExit(1)
    stream_b = low_latency_capture(ResilientCapture(
        source_b,
        "B",
        cap_b,
        args.open_attempts,
        max_reconnects=max(0, args.max_reconnects),
    ))

    print("Dang chay tracking Global ID. Nhan 'q' hoac ESC de thoat.")
    # Voice Ha Linh BAN CAMERA luon bat song song tracking (subprocess
    # rieng, tu restart khi crash). Mic RTSP camera + loa camera P2P
    # (doi lap voi ban _local dung mic/loa laptop). Tat bang --no-halinh.
    # Hello imou + STT small da bo: chi medium.
    halinh_stop: threading.Event | None = None
    halinh_thread: threading.Thread | None = None
    if args.halinh:
        halinh_script = (args.halinh_script
                         or str(_PROJECT_ROOT / "scripts" / "halinh_assistant.py"))
        if not Path(halinh_script).is_file():
            print(f"[HaLinh-sup] khong thay {halinh_script}, bo qua voice.")
        else:
            halinh_extra = ["--wake-model", str(args.halinh_wake_model),
                              "--cmd-model", str(args.halinh_cmd_model)]
            if args.halinh_channel is not None:
                halinh_extra += ["--channel", str(args.halinh_channel)]
            halinh_stop = threading.Event()
            halinh_thread = threading.Thread(
                target=supervise_halinh,
                args=(halinh_script, halinh_stop, args.halinh_restarts,
                      halinh_extra),
                name="halinh-supervisor",
                daemon=True,
            )
            halinh_thread.start()
    start = time.time()
    metrics = StageMetrics()
    frame_idx = 0
    last_tracks_a = []
    last_tracks_b = []
    last_states_a: dict[int, str] = {}
    last_states_b: dict[int, str] = {}
    window_a_shown = False
    window_b_shown = False
    seen_gids: set[int] = set()
    ghosts_dir = Path(config.output.output_dir) / "ghosts"
    face_marks_a: list = []  # (x1,y1,x2,y2,score,known) for cam A overlay
    face_marks_b: list = []  # (x1,y1,x2,y2,score,known) for cam B overlay
    frame_hub = FrameHub()
    from camera_tracking.evaluation import IdentityTraceWriter
    identity_trace = IdentityTraceWriter(args.identity_log)
    # Voice vay tay: mac dinh BAT (tat bang --no-greet); TTS cache + loa camera.
    # backend mac dinh moi imou_p2p (P2P VisualTalk, khong browser);
    # imou_web giu lai legacy, local = ffplay tai may chay pipeline.
    voice_cfg = config.voice
    foreground_selectors = {
        channel: ForegroundSelector(
            switch_area_ratio=float(getattr(
                voice_cfg, "focus_switch_area_ratio", 1.20)),
            switch_hold_s=float(getattr(
                voice_cfg, "focus_switch_hold_s", 0.50)),
            lost_grace_s=float(getattr(
                voice_cfg, "focus_lost_grace_s", 1.00)),
        )
        for channel in ("A", "B")
    }
    foreground_by_channel: dict[str, int | None] = {"A": None, "B": None}
    voice_on = bool(args.greet or voice_cfg.enabled)
    wave_detector = palm_detector = greeter = None
    gesture_detectors: dict[str, object] = {}
    gesture_modes: dict[str, str] = {
        "A": str(getattr(voice_cfg, "gesture_a", "palm")),
        "B": str(getattr(voice_cfg, "gesture_b", "palm")),
    }
    voice_bridge = None
    voice_tick = 0
    wave_hint_at: dict[int, float] = {}
    # Throttle log chan doan gesture (5s/gid) de khong spam console.
    palm_dbg_at: dict[int, float] = {}
    if voice_on:
        try:
            from camera_tracking.gesture.wave import (
                MediaPipeHandDetector,
                OpenPalmDetector,
                WaveDetector,
            )
            from camera_tracking.voice.greeter import VoiceGreeter

            _need_hand = any(m in ("wave", "palm") for m in gesture_modes.values())
            hand_detector = None
            if _need_hand:
                hand_detector = MediaPipeHandDetector(
                    min_detection_confidence=float(getattr(
                        voice_cfg, "palm_min_detection_confidence", 0.6)),
                    min_input_height_px=int(getattr(
                        voice_cfg, "palm_min_input_height_px", 0)),
                )
                if not hand_detector.available:
                    raise RuntimeError("mediapipe hands unavailable")
                # Tai hand_landmarker.task lan dau ngay tai startup (loi mang
                # hien ro o day thay vi treo frame xu ly dau tien).
                hand_detector.ensure_loaded()
            # A=phong: vay tay (WaveDetector) de chao; B=cua: off (face-trigger).
            # Giay phep legacy palm qua config gesture_*=palm.
            _palm_cooldown = float(getattr(
                voice_cfg, "palm_cooldown_s",
                getattr(voice_cfg, "wave_cooldown_s", 30.0)))
            _unknown_cooldown = float(getattr(
                voice_cfg, "unknown_cooldown_s", 10.0))
            _wave_cooldown = float(getattr(voice_cfg, "wave_cooldown_s", 10.0))
            for _ch, _mode in gesture_modes.items():
                if _mode == "wave" and hand_detector is not None:
                    gesture_detectors[_ch] = WaveDetector(
                        window_s=float(getattr(voice_cfg, "wave_window_s", 2.5)),
                        min_reversals=int(getattr(
                            voice_cfg, "wave_min_reversals", 2)),
                        min_amplitude=float(getattr(
                            voice_cfg, "wave_min_amplitude", 0.06)),
                        min_gap_s=float(getattr(
                            voice_cfg, "wave_min_gap_s", 0.08)),
                        cooldown_s=min(_wave_cooldown, _unknown_cooldown),
                        required_fingers=float(getattr(
                            voice_cfg, "wave_required_fingers", 4.5)),
                        palm_confirm_frames=int(getattr(
                            voice_cfg, "wave_palm_confirm_frames", 2)),
                        palm_release_frames=int(getattr(
                            voice_cfg, "wave_palm_release_frames", 5)),
                        hand_detector=hand_detector,
                    )
                elif _mode == "palm" and hand_detector is not None:
                    gesture_detectors[_ch] = OpenPalmDetector(
                        required_fingers=float(getattr(
                            voice_cfg, "palm_required_fingers", 4)),
                        cooldown_s=min(_palm_cooldown, _unknown_cooldown),
                        confirm_frames=int(getattr(
                            voice_cfg, "palm_confirm_frames", 1)),
                        release_frames=int(getattr(
                            voice_cfg, "palm_release_frames", 2)),
                        hand_detector=hand_detector,
                        max_hand_center_y=float(getattr(
                            voice_cfg, "palm_head_region_max_y", 0.60)),
                    )
            # Tuong thich code cu dung bien don: uu tien detector cua A,
            # fallback detector dau tien con lai.
            if gesture_detectors:
                _first = next(iter(gesture_detectors.values()))
                wave_detector = gesture_detectors.get("A", _first)
                # palm_detector giu de cac doan forget_retired/loop cu van chay;
                # thuc te loop moi dung gesture_detectors theo kenh.
                palm_detector = _first
            voice_output = None
            if voice_cfg.backend == "imou_p2p":
                from camera_tracking.voice.p2p_talk import (
                    ImouP2PCredentials,
                    ImouP2PTalkOutput,
                )

                voice_output = ImouP2PTalkOutput(
                    ImouP2PCredentials.from_env(),
                    channel=voice_cfg.p2p_channel,
                    timeout=voice_cfg.p2p_timeout_s,
                    attempts=voice_cfg.p2p_attempts,
                    retry_delay=voice_cfg.p2p_retry_delay_s,
                    sample_rate=voice_cfg.p2p_sample_rate,
                    volume=voice_cfg.p2p_volume,
                    bind_port=int(getattr(voice_cfg, 'p2p_bind_port', 18086)),
                )
            elif voice_cfg.backend == "imou_web":
                from camera_tracking.voice.imou_bridge import (
                    ImouAudioTalkBridge,
                    ImouAudioTalkOutput,
                )

                voice_bridge = ImouAudioTalkBridge.from_env(
                    host=voice_cfg.bridge_host,
                    port=voice_cfg.bridge_port,
                    command_ttl_s=voice_cfg.command_ttl_s,
                    talk_tail_s=voice_cfg.talk_tail_s,
                    launch_browser=voice_cfg.launch_browser,
                )
                voice_bridge.start()
                voice_output = ImouAudioTalkOutput(voice_bridge)
            # WAV tao san (ZeroTTS, 1 cau = 1 file): khop nguyen van cau
            # chao thi dung ngay, khong can mang/TTS runtime.
            # Moi cau chao la 1 ban thu hoan chinh theo dung ten hien thi
            # ("Xin chào Anh Duy", khong ghep noi): khop manifest thi dung
            # file san, khong thi tong hop edge-tts luc chay (co cache).
            from camera_tracking.voice.zerotts_tts import load_phrase_files

            phrase_files = load_phrase_files(voice_cfg.greeting_dir)
            if not phrase_files:
                print("Voice greetings manifest trong; "
                      "chay scripts/build_greeting_wavs.py de tao san.")
            greeter = VoiceGreeter(
                cache_dir=voice_cfg.cache_dir,
                voice=voice_cfg.voice,
                cooldown_s=voice_cfg.cooldown_s,
                unknown_cooldown_s=float(getattr(
                    voice_cfg, "unknown_cooldown_s", 10.0)),
                unknown_phrase=voice_cfg.unknown_phrase,
                max_queue_age_s=voice_cfg.command_ttl_s,
                proactive_quiet_s=float(getattr(
                    voice_cfg, "proactive_quiet_s", 5.0)),
                output=voice_output,
                phrase_files=phrase_files,
            )
            greeter.start()
            # Mo san duong truyen loa o nen: lan chao dau khong mat tien
            # bat tay cloud (tu ~10s xuong con vai giay).
            try:
                _warm = getattr(voice_output, "warmup", None)
                if callable(_warm):
                    threading.Thread(
                        target=_warm, name="p2p-warmup", daemon=True).start()
            except (AttributeError, RuntimeError):
                pass
            phrases = [voice_cfg.unknown_phrase]
            if face_gallery is not None:
                phrases.extend(
                    f"Xin chào {person.display_name}" for person in face_gallery.people
                )
            greeter.prewarm(phrases)
            print(f"Voice: {voice_cfg.backend}, greet on wave ({voice_cfg.voice}, "
                  f"cooldown {voice_cfg.cooldown_s:.0f}s quen / "
                  f"{float(getattr(voice_cfg, 'unknown_cooldown_s', 10.0)):.0f}s la, TTL "
                  f"{voice_cfg.command_ttl_s:.0f}s, pregen={len(phrase_files)}).")
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            print(f"Voice disabled ({error})")
            voice_on = False
            wave_detector = palm_detector = greeter = None
            gesture_detectors.clear()
            if voice_bridge is not None:
                voice_bridge.close()
                voice_bridge = None
    # Voice-trigger "hello imou" DA BO (user khong dung): chi giu greet
    # vay tay/palm + face-trigger. Khong mo mic RTSP, khong STT small,
    # khong inhibit 12s. Flag --voice-trigger giu de tuong thich CLI
    # nhung khong con tac dung.
    voice_trigger = None
    voice_listener = None
    voice_stop = None
    trigger_on = False
    # A=phong: wave-only (khong chao mat); B=cua: face-trigger (thay mat la chao).
    # Cau hinh per-channel (greet_on_face_{a,b}) fallback ve global de tuong
    # thich cu; --no-face-greet tat het.
    def _face_greet_enabled(_ch: str) -> bool:
        if not voice_on or greeter is None or args.no_face_greet:
            return False
        _per = getattr(voice_cfg, f"greet_on_face_{_ch.lower()}", None)
        _base = voice_cfg.greet_on_face if _per is None else bool(_per)
        return bool(_base)

    def _unknown_face_greet_enabled(_ch: str) -> bool:
        _per = getattr(voice_cfg, f"greet_unknown_on_face_{_ch.lower()}", None)
        _base = voice_cfg.greet_unknown_on_face if _per is None else bool(_per)
        return bool(_base)

    def _speaker_channel(_ch: str) -> int:
        """Loa P2P theo camera phat hien: A (phong) -> p2p_channel,
        B (cua) -> p2p_channel_b."""
        if str(_ch).upper() == "B":
            try:
                return int(getattr(voice_cfg, "p2p_channel_b",
                                   voice_cfg.p2p_channel))
            except (AttributeError, TypeError, ValueError):
                return int(voice_cfg.p2p_channel)
        return int(voice_cfg.p2p_channel)

    def _ha_turn_active() -> bool:
        """True khi Ha Linh dang hoi-dap -> chao nhuong turn, khoi noi chong.

        Doc file khoa TTL (tu het han neu subprocess crash). Voice khong
        bao gio duoc lam chet pipeline nen moi loi deu -> False.
        """
        try:
            from camera_tracking.voice.turn_guard import turn_active
            return bool(turn_active())
        except Exception:  # noqa: BLE001
            return False

    face_greet_on_by_channel: dict[str, bool] = {
        _ch: _face_greet_enabled(_ch) for _ch in ("A", "B")
    }
    # Tuong thich code cu dung bien don (True neu bat ky kenh nao bat).
    face_greet_on = bool(any(face_greet_on_by_channel.values()))
    if voice_on and not face_greet_on:
        if trigger_on:
            print("Voice: A vẫy tay HOẶC nói cụm gọi (wave OR voice).")
        else:
            print("Voice: A chi chao khi vay tay (wave-only).")
    elif face_greet_on:
        _on = [c for c, v in face_greet_on_by_channel.items() if v]
        print(f"Voice: chao tu dong khi nhan dien mat o kenh {','.join(_on)} "
              f"(A wave-only, B face-trigger).")
    # Live event feed cho dashboard (kiosk + admin): 20 su kien moi nhat,
    # song song voi WriteQueue -> Supabase (kiosk thay ngay ca khi offline).
    recent_events: deque = deque(maxlen=20)

    def _note_event(*, event: str, global_id: int, channel: str,
                    at_iso: str, person_id=None, person_name=None) -> None:
        recent_events.append({
            "event": event,
            "global_id": global_id,
            "channel": channel,
            "at": at_iso,
            "person_id": person_id,
            "person_name": person_name,
        })

    def _handle_one_face(*, channel: str, frame, track, crop, det, match,
                           sharp, quality: float = 1.0,
                           identity_confirmed: bool = True,
                           face_marks: list, day_str: str,
                           wall_iso: str, time_tag: str, now_s: float) -> None:
        """Apply 1 face observation: bind/reconcile/attendance/greet/crop-save.

        Dùng chung cho cả sync fallback và async worker (FaceResult).
        """
        nonlocal unknown_counter
        if frame is not None:
            _bh, _bw = frame.shape[:2]
        else:
            # Ket qua async van co gia tri cho matching/attendance khi RTSP
            # vua hut dung mot frame. Chi can kich thuoc du de doi bbox crop
            # ve toa do full-frame; khong bo ca ket qua da inference xong.
            _bh = max(1, int(track.bbox.y2) + 1)
            _bw = max(1, int(track.bbox.x2) + 1)
        _ox = max(0, min(_bw, round(track.bbox.x1)))
        _oy = max(0, min(_bh, round(track.bbox.y1)))
        fx1 = max(0, int(det.bbox[0])); fy1 = max(0, int(det.bbox[1]))
        try:
            _cw = crop.shape[1]
        except (AttributeError, IndexError):
            _cw = fx1 + 1
        fx2 = min(_cw, int(det.bbox[2]))
        try:
            _ch = crop.shape[0]
        except (AttributeError, IndexError):
            _ch = fy1 + 1
        fy2 = min(_ch, int(det.bbox[3]))
        raw_gid = track.track_id
        face_box = (float(fx1), float(fy1), float(fx2), float(fy2))
        # Store normalized coordinates so the gate follows the same person
        # while their tracking bbox changes size or position. Pose is not a
        # gate: a clearly detected side-looking face remains eligible.
        _crop_w = max(1.0, float(_cw))
        _crop_h = max(1.0, float(_ch))
        _norm_face = (
            face_box[0] / _crop_w, face_box[1] / _crop_h,
            face_box[2] / _crop_w, face_box[3] / _crop_h,
        )
        recent_face_by_track[(channel, raw_gid)] = (now_s, _norm_face)
        face_quality_by_track[(channel, raw_gid)] = (float(quality), now_s)
        if face_gallery is not None:
            ranked = face_gallery.ranked_matches(det.embedding)[:2]
            gid_to_face_candidates[raw_gid] = [
                (person.display_name, float(score)) for person, score in ranked
            ]
            # Only a ranking whose own Top-1 passed threshold + ambiguity
            # margin may participate. Top-2 is a fallback solely when Top-1
            # was taken by another, higher-scoring visible face.
            if match.is_known:
                gid_to_face_rankings[raw_gid] = list(ranked)
                gid_face_ranking_at[raw_gid] = now_s

            _ranking_ttl = max(
                3.0, float(getattr(face_cfg, "consensus_window_s", 3.0)) + 2.0)
            _alive = set(manager.identities)
            _fresh_rankings = {
                gid: candidates
                for gid, candidates in gid_to_face_rankings.items()
                if gid in _alive
                and now_s - gid_face_ranking_at.get(gid, float("-inf"))
                <= _ranking_ttl
            }
            _assignments = unique_face_assignments(
                _fresh_rankings, threshold=float(face_threshold), limit=2)

            # A live GID that lost a score conflict must immediately lose the
            # stale bbox name. Never merge two simultaneously visible people
            # merely because their Top-1 gallery candidate was identical.
            for _ranked_gid in _fresh_rankings:
                _chosen = _assignments.get(_ranked_gid)
                _chosen_employee = (
                    _chosen.employee_id if _chosen is not None else None)
                _old_employee = gid_to_person.get(_ranked_gid)
                if _old_employee and _old_employee != _chosen_employee:
                    manager.unbind_employee(_ranked_gid)
                    gid_to_person.pop(_ranked_gid, None)
                    gid_to_display.pop(_ranked_gid, None)
                    gid_to_score.pop(_ranked_gid, None)

            _assigned = _assignments.get(raw_gid)
            if _assigned is None:
                match = MatchResult(
                    person=None,
                    score=float(ranked[0][1]) if ranked else 0.0,
                    is_known=False,
                    second_score=(float(ranked[1][1]) if len(ranked) > 1 else 0.0),
                    margin=(float(ranked[0][1] - ranked[1][1])
                            if len(ranked) > 1 else 0.0),
                )
            else:
                match = MatchResult(
                    person=_assigned.person,
                    score=float(_assigned.score),
                    is_known=True,
                    second_score=(float(ranked[1][1]) if len(ranked) > 1 else 0.0),
                    margin=(float(ranked[0][1] - ranked[1][1])
                            if len(ranked) > 1 else 0.0),
                )
                # Confirm the *resolved* candidate, not the independent local
                # Top-1. Requiring a repeated resolved result gives other
                # visible faces time to enter the same score competition
                # before attendance/greeting side effects are allowed.
                _assignment_vote = _candidate_vote_reached(
                    face_assignment_streaks, raw_gid,
                    _assigned.employee_id, float(_assigned.score), now_s,
                    required_hits=max(2, int(face_cfg.consensus_hits)),
                    window_s=max(
                        3.0, float(face_cfg.consensus_window_s) + 2.0),
                    min_score=float(face_threshold),
                )
                identity_confirmed = _assignment_vote is not None

        # Live feedback uses the collision-resolved match, not the independent
        # per-face Top-1 result.
        face_marks.append(
            (_ox + fx1, _oy + fy1, _ox + fx2, _oy + fy2,
             match.score, match.is_known, track.track_id))
        # Lan quan sat dau tien van cap nhat box + top-2 len man hinh. Cac
        # side effect nhay cam (bind employee, diem danh, greet, save crop)
        # chi chay sau khi dat consensus.
        if not identity_confirmed:
            return
        if match.is_known and match.person is not None:
            employee_id = (
                match.person.employee_id
                or match.person.person_id
            )
            identity_gid = track.track_id
            # The joint score resolver is authoritative. If an older GID was
            # provisionally holding this name, release it; do not merge two
            # people who are simultaneously visible in the same room.
            conflict_gid = next(
                (gid for gid, record in manager.identities.items()
                 if gid != identity_gid
                 and gid not in gid_alias
                 and record.employee_id == employee_id),
                None,
            )
            if conflict_gid is not None:
                manager.unbind_employee(conflict_gid)
                gid_to_person.pop(conflict_gid, None)
                gid_to_display.pop(conflict_gid, None)
                gid_to_score.pop(conflict_gid, None)
            _current_employee = manager.employee_id_of(identity_gid)
            if _current_employee not in (None, employee_id):
                manager.unbind_employee(identity_gid)
                gid_to_person.pop(identity_gid, None)
                gid_to_display.pop(identity_gid, None)
                gid_to_score.pop(identity_gid, None)
            if not manager.bind_employee(identity_gid, employee_id):
                print(f"[Identity conflict] refused {employee_id} -> G{identity_gid}")
                return
            # Face anchor: GID vua duoc face diem cao xac nhan -> khoa tam
            # thoi de chong B cuop khi che A hoan toan (xem global_identity).
            try:
                manager.set_face_anchor(
                    identity_gid, employee_id, float(match.score), now_s)
            except (AttributeError, TypeError, ValueError):
                pass
            if face_worker is not None:
                face_worker.mark_known(identity_gid, now_s, channel)
            gid_to_score[identity_gid] = max(
                match.score, gid_to_score.get(identity_gid, 0.0)
            )
            gid_to_person[identity_gid] = employee_id
            # Show the name as soon as the face matches
            # (DB tick stays debounced below).
            gid_to_display[identity_gid] = \
                match.person.display_name
            recent_face_by_track[(channel, identity_gid)] = (now_s, _norm_face)
            face_quality_by_track[(channel, identity_gid)] = (
                float(quality), now_s)
            # Da bind chinh thuc: xoa streak provisional (tranh merge lap).
            face_candidate_streaks.pop(identity_gid, None)
            # Face-triggered greeting per-channel: B=cua thay mat la chao
            # ngay (khong can tay); A=phong wave-only nen tat.
            # Chung cooldown/queue voi gesture de dung lau khong spam.
            # Da dinh danh: huy hen/luot cho "quy khach" cua GID de chi
            # chao ten, khong noi 2 cau lien tiep.
            try:
                pending_unknown_greet.pop(track.track_id, None)
                pending_unknown_greet.pop(identity_gid, None)
                if greeter is not None:
                    greeter.cancel_pending_unknown(global_id=track.track_id)
                    if identity_gid != track.track_id:
                        greeter.cancel_pending_unknown(global_id=identity_gid)
            except Exception:  # noqa: BLE001 - voice khong chet pipeline
                pass
            _focus_gid = foreground_by_channel.get(channel)
            _is_foreground = _focus_gid in (raw_gid, identity_gid)
            if (face_greet_on_by_channel.get(channel, face_greet_on)
                    and greeter is not None and _is_foreground
                    and not _ha_turn_active()):
                try:
                    _spk = _speaker_channel(channel)
                    _guest_age: float | None = None
                    try:
                        _guest_age = greeter.unknown_spoken_age(
                            day=day_str, global_id=track.track_id,
                            channel=_spk, now_s=now_s)
                        if identity_gid != track.track_id:
                            _age2 = greeter.unknown_spoken_age(
                                day=day_str, global_id=identity_gid,
                                channel=_spk, now_s=now_s)
                            if _age2 is not None and (
                                _guest_age is None or _age2 < _guest_age
                            ):
                                _guest_age = _age2
                    except (AttributeError, TypeError, ValueError):
                        _guest_age = None
                    _guest_window = float(getattr(
                        greeter, "unknown_cooldown_s", 10.0))
                    if _guest_age is not None and _guest_age < _guest_window:
                        print(f"[Voice][{channel}] "
                              f"{match.person.display_name} vua duoc chao khach "
                              f"{_guest_age:.0f}s truoc, bo chao ten")
                    elif greeter.face_greet(
                        day=day_str, person_id=employee_id,
                        display_name=match.person.display_name,
                        now_s=now_s, global_id=identity_gid,
                        channel=_spk,
                    ):
                        _note_event(
                            event="FACE_GREET", global_id=identity_gid,
                            channel=channel, at_iso=wall_iso,
                            person_id=employee_id,
                            person_name=match.person.display_name,
                        )
                        print(f"[Voice][{channel}] Chao "
                              f"{match.person.display_name} (G{identity_gid}, "
                              f"loa ch{_spk})")
                except Exception as error:  # noqa: BLE001 - voice khong chet pipeline
                    print(f"[Voice] Bo loi chao mat: {error}")
            # Chỉ grow gallery từ quan sát chất lượng cao để chống
            # prototype pollution (ảnh mờ/góc xấu làm bẩn gallery).
            if (
                quality >= 0.25
                and sharp >= 40.0
                and face_gallery.add_prototype(
                    match.person.person_id,
                    det.embedding,
                    seed_similarity=match.score,
                    min_seed_similarity=face_cfg.gallery_accept_threshold,
                    max_prototypes=face_cfg.gallery_max_prototypes,
                )
            ):
                save_prototype(
                    face_cfg.gallery_dir,
                    match.person.person_id,
                    det.embedding,
                    index=len(match.person.prototypes),
                )
            ticked = attendance.observe(
                day=day_str, global_id=identity_gid,
                person_id=employee_id,
                display_name=match.person.display_name,
                score=match.score, now_s=now_s,
                wall_time_iso=wall_iso,
            )
            if (
                ticked is not None
                and day_cache.attendance_should_write(day_str, ticked.person_id)
            ):
                    _store_or_queue(write_queue, supabase, "attendance", {
                        "date": day_str, "person_id": ticked.person_id,
                        "person_name": ticked.display_name,
                        "global_id": ticked.global_id,
                        "attended": True, "check_in_at": wall_iso,
                        "face_score": ticked.face_score,
                        # Tick biên (vừa đủ ngưỡng) cần admin review tay.
                        "needs_review": bool(ticked.face_score < 0.80),
                    })
                    _store_or_queue(write_queue, supabase, "event", {
                        "date": day_str, "global_id": ticked.global_id,
                        "person_id": ticked.person_id,
                        "event": "CHECK_IN", "channel": channel,
                        "at": wall_iso,
                    })
                    _note_event(
                        event="CHECK_IN", global_id=ticked.global_id,
                        channel=channel, at_iso=wall_iso,
                        person_id=ticked.person_id,
                        person_name=ticked.display_name,
                    )
                    print(f"[Diem danh][{channel}] {ticked.display_name} "
                          f"(G{ticked.global_id}, "
                          f"score={ticked.face_score:.2f})")
            # Reconcile: stale unknown GIDs that match
            # this person merge into the current gid.
            if match.person.embedding is not None:
                _reconcile_unknowns(
                    day_cache, write_queue, supabase,
                    reconciler, gid_alias, gid_to_person,
                    gid_to_display, unknown_of_gid,
                    day_str, identity_gid,
                    employee_id,
                    match.person.display_name,
                    match.person.embedding,
                    (business_a, business_b),
                )
            # Same face on a stale duplicate gid:
            # fold it into the live one (fixes the
            # G1-OUT + G2-Working split-brain).
            _merge_same_person_stale(
                day_cache, write_queue, supabase,
                gid_alias, gid_to_person,
                gid_to_display, day_str,
                identity_gid,
                employee_id,
                match.person.display_name,
                manager, (business_a, business_b),
                now_s=now_s, channel=channel,
            )
            local = crop_saver.save_best_crop(
                day=day_str, owner=employee_id,
                known=True, global_id=identity_gid,
                image_bgr=crop, face_score=match.score,
                sharpness=sharp, time_tag=time_tag,
            )
            if local:
                _store_or_queue(write_queue, supabase, "face_crop", {
                    "local_path": local,
                    "storage_path": f"{day_str}/known/{Path(local).name}",
                })
        else:
            if track.track_id not in unknown_of_gid:
                unknown_counter += 1
                unknown_of_gid[track.track_id] = (
                    f"U-{day_str.replace('-', '')}-{unknown_counter:03d}"
                )
            uid = unknown_of_gid[track.track_id]
            reconciler.note_unknown(
                uid, det.embedding, match.score)
            if (
                face_greet_on_by_channel.get(channel, face_greet_on)
                and greeter is not None
                and _unknown_face_greet_enabled(channel)
                and foreground_by_channel.get(channel) == track.track_id
            ):
                # Chao khach la HOAN: cho dinh danh unknown_greet_delay_s
                # truoc khi chao "quy khach". Mat xac nhan kip thi nhanh
                # known o tren da huy hen -> chi chao ten, khong 2 cau.
                try:
                    _ug_delay = max(0.0, float(getattr(
                        voice_cfg, "unknown_greet_delay_s", 2.0)))
                    _ug_gid = track.track_id
                    if (
                        gid_to_person.get(_ug_gid) is None
                        and _ug_gid not in gid_alias
                        and _ug_gid not in pending_unknown_greet
                    ):
                        pending_unknown_greet[_ug_gid] = (
                            now_s + _ug_delay, day_str, channel,
                            wall_iso, _speaker_channel(channel),
                        )
                except Exception as error:  # noqa: BLE001
                    print(f"[Voice] Bo loi hen chao khach: {error}")
            local = crop_saver.save_best_crop(
                day=day_str, owner=uid, known=False,
                global_id=track.track_id, image_bgr=crop,
                face_score=match.score, sharpness=sharp,
                time_tag=time_tag,
            )
            if local:
                _store_or_queue(write_queue, supabase, "face_crop", {
                    "local_path": local,
                    "storage_path":
                        f"{day_str}/unknown/{Path(local).name}",
                })

    def _process_face_channel(*, channel: str, frame, confirmed_tracks,
                              face_marks: list, day_str: str,
                              wall_iso: str, time_tag: str,
                              now_s: float) -> None:
        """Sync fallback khi không có FaceWorker (giữ hành vi cũ)."""
        if not (use_face and attendance is not None
                and face_matcher is not None and face_consumer is not None):
            return
        if channel not in face_consumer.channels:
            return
        event = TrackEvent(
            channel=channel,
            frame=Frame(frame_idx, now_s, frame),
            tracks=tuple(confirmed_tracks),
        )
        with metrics.measure("face"), face_model_lock:
            observations = face_consumer.consume(event, now_s)
        for observation in observations:
            _handle_one_face(
                channel=channel, frame=frame, track=observation.track,
                crop=observation.crop_bgr, det=observation.detection,
                match=observation.match, sharp=observation.sharpness,
                quality=observation.quality, face_marks=face_marks,
                identity_confirmed=observation.identity_confirmed,
                day_str=day_str, wall_iso=wall_iso, time_tag=time_tag,
                now_s=now_s,
            )

    def _enqueue_face_async(*, channel: str, frame, confirmed_tracks,
                            day_str: str, wall_iso: str, time_tag: str,
                            now_s: float) -> None:
        """Event-driven once-per-track: chỉ enqueue khi thật sự cần.

        - Đã biết tên -> recheck sau face.known_cooldown_s (mặc định 30s).
        - Chưa biết -> thử lại sau face.unknown_cooldown_s (~0.6s) khi góc đẹp hơn.
        - Backpressure theo khả năng máy: queue càng đầy thì cooldown unknown
          càng giãn (x1.0 khi rảnh -> x2.0 khi đầy) để máy yếu không bị dí GPU.
        - Bỏ qua track nhỏ (không đủ pixel mặt) ngay ở main loop (rẻ).
        Main loop không bao giờ chờ inference ở đây.
        Production vẫn vẽ box mặt mỗi khi có FaceResult tươi (xem vis bên dưới).
        """
        if not (use_face and face_worker is not None
                and attendance is not None and face_consumer is not None):
            return
        if channel not in face_consumer.channels:
            return
        if not face_worker.ensure_alive():
            print("[FaceWorker] worker không chạy; bỏ enqueue để tránh treo ngầm")
            return
        try:
            _load = float(face_worker.load_ratio)
        except (AttributeError, TypeError, ValueError):
            _load = 0.0
        if _load >= 1.0:
            # Máy đang quá tải: bỏ lượt này, frame sau queue vơi sẽ thử lại.
            return
        _base_recheck = min(
            float(getattr(config.voice, "face_recheck_s", 30.0)),
            float(getattr(face_cfg, "known_cooldown_s", 30.0)),
        )
        recheck_s = _base_recheck
        if voice_on:
            # Greeting can face+hand cung luc, nen can mot face observation moi.
            recheck_s = min(recheck_s, 2.0)
        _base_unknown = float(getattr(
            face_consumer, "unknown_cooldown_s", 0.6))
        # May khoe (queue ranh): giu 0.6s. May yeu (queue day): gian toi 1.2s.
        _eff_unknown = _base_unknown * (1.0 + max(0.0, min(1.0, _load)))
        focus_gid = foreground_by_channel.get(channel)
        aging_s = float(getattr(config.voice, "face_background_aging_s", 5.0))

        def _face_priority(track) -> tuple[float, float, float]:
            known = track.track_id in gid_to_person
            if track.track_id == focus_gid and not known:
                tier = 0.0
            elif not known:
                tier = 1.0
            else:
                tier = 2.0
            cached = face_quality_by_track.get((channel, track.track_id))
            quality = float(cached[0]) if cached is not None else 0.0
            age_bonus = (max(0.0, now_s - cached[1]) / max(0.1, aging_s)
                         if cached is not None else 0.0)
            return (tier, -(quality + min(1.0, age_bonus)),
                    -float(track.bbox.area))

        for track in sorted(confirmed_tracks, key=_face_priority):
            gid = track.track_id
            if track.bbox.area < face_cfg.min_person_area_px:
                continue
            known = gid in gid_to_person
            if not face_worker.need_face(
                gid, known=known, now_s=now_s, recheck_s=recheck_s,
                unknown_cooldown_s=_eff_unknown,
                channel=channel,
            ):
                continue
            crop = _person_crop(frame, track.bbox)
            if crop is None or crop.size == 0:
                continue
            try:
                from camera_tracking.face.worker import FaceJob as _FJ
                accepted = face_worker.submit(_FJ(
                    channel=channel, gid=gid, crop_bgr=crop.copy(),
                    track=track, day_str=day_str, wall_iso=wall_iso,
                    time_tag=time_tag, now_s=now_s,
                    priority=_face_priority(track),
                ))
                if accepted:
                    face_worker.mark_attempt(gid, now_s, channel)
            except Exception as error:
                print(f"[FaceWorker] enqueue {channel}/G{gid} lỗi: {error}")
                continue

    def _drain_face_results(*, frame_a, frame_b, face_marks_a: list,
                            face_marks_b: list) -> None:
        """Poll worker (không block) rồi apply bind/attendance/greet."""
        if face_worker is None:
            return
        try:
            results = face_worker.poll_results()
        except Exception as error:
            print(f"[FaceWorker] poll result lỗi: {error}")
            return
        if not results:
            return
        marks_by_channel = {"A": face_marks_a, "B": face_marks_b}
        frames_by_channel = {"A": frame_a, "B": frame_b}
        with metrics.measure("face_apply"):
            for res in results:
                frame = frames_by_channel.get(res.channel)
                marks = marks_by_channel.get(res.channel)
                if marks is None:
                    continue
                try:
                    _handle_one_face(
                        channel=res.channel, frame=frame, track=res.track,
                        crop=res.crop_bgr, det=res.detection,
                        match=res.match, sharp=res.sharpness,
                        quality=getattr(res, "quality", 1.0),
                        identity_confirmed=getattr(
                            res, "identity_confirmed", True
                        ),
                        face_marks=marks, day_str=res.day_str,
                        wall_iso=res.wall_iso, time_tag=res.time_tag,
                        now_s=res.now_s,
                    )
                except Exception as error:  # noqa: BLE001
                    print(f"[Face] apply skipped G{res.gid}: {error}")
                    continue

    def poll_voice_command() -> None:
        """Doc lenh voice (chao/chup) va thuc thi, ghi ket qua lai file."""
        import json as _json

        try:
            target = (_PROJECT_ROOT / "output" / "qa_cache"
                      / "voice_command.json")
            cmd = _json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(cmd, dict) or cmd.get("status") != "pending":
            return
        try:
            age = time.time() - float(cmd.get("ts", 0))
        except (TypeError, ValueError):
            age = 9999.0
        if age > 60:
            cmd["status"] = "expired"
        else:
            def _find_person(name: str):
                want = name.strip().lower()
                for gid, disp in gid_to_display.items():
                    if not disp:
                        continue
                    have = disp.strip().lower()
                    if have and (have in want or want in have):
                        return gid_to_person.get(gid), disp
                return None

            def _save_snapshot(camera: str | None):
                import datetime as _dt

                use_b = (camera or "A").strip().upper() == "B"
                frame = (vis_b if use_b else vis_a)
                if frame is None:
                    frame = frame_b if use_b else frame_a
                if frame is None:
                    return None
                snap_dir = _PROJECT_ROOT / "output" / "snapshots"
                try:
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                    path = (snap_dir /
                            f"snap_cam_{(camera or 'A').strip().upper()}_{stamp}.jpg")
                    if not cv2.imwrite(str(path), frame):
                        return None
                except (OSError, cv2.error):
                    return None
                return str(path.relative_to(_PROJECT_ROOT))

            ok, message = execute_voice_command(
                cmd, greeter=greeter, find_person=_find_person,
                save_snapshot=_save_snapshot, day=day_str, now_s=now_s)
            cmd["status"] = "done" if ok else "failed"
            cmd["message"] = message
            print(f"[Voice-cmd] {cmd.get('action')}: {message}", flush=True)
        try:
            tmp = target.with_suffix(".tmp")
            tmp.write_text(_json.dumps(cmd, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(target)
        except OSError:
            pass

    def write_runtime_status() -> None:
        """Ghi trang thai runtime cho voice Ha Linh BAN CAMERA doc (2s/lan).

        File: output/qa_cache/runtime_status.json. Voice tool
        (get_person_count, get_camera_status...) doc file nay de tra loi
        so lieu THAT thay vi stub random. Loi thi bo qua (khong chet loop).
        """
        import json as _json

        now_wall = time.time()
        people = []
        try:
            for gid, st in room_status_now.items():
                wgid = gid_alias.get(gid, gid)
                record = manager.identities.get(wgid)
                last_s = record.last_seen_s if record is not None else None
                ago = (now_s - last_s) if last_s is not None else None
                people.append({
                    "gid": int(wgid),
                    "name": str(getattr(st, "display_name", "") or ""),
                    "label": str(getattr(st, "label", "") or ""),
                    "in_room": bool(getattr(st, "in_room", False)),
                    "last_seen_ago_s": ago,
                })
        except Exception:
            people = []
        try:
            day = day_str
            att = ([{"name": r.display_name, "time": r.wall_time}
                    for r in attendance.records_for_day(day)]
                   if attendance is not None else [])
        except Exception:
            att = []
        try:
            events = []
            for item in list(recent_events)[-5:]:
                events.append(item.get("event", str(item))
                              if isinstance(item, dict) else str(item))
        except Exception:
            events = []
        try:
            det = (metrics.snapshot() or {}).get("detection", {}) or {}
            infer_ms = det.get("mean_ms")
        except Exception:
            infer_ms = None
        prev_t, prev_f = runtime_prev["t"], runtime_prev["frame"]
        dt = now_wall - prev_t
        fps = (frame_idx - prev_f) / dt if dt > 0.5 else None
        runtime_prev["t"], runtime_prev["frame"] = now_wall, frame_idx

        def _stream_state(stream, enabled: bool) -> dict:
            inner = getattr(stream, "stream", stream)
            capture = getattr(inner, "capture", None)
            try:
                opened = bool(capture.isOpened()) if capture is not None else False
            except Exception:
                opened = False
            try:
                exhausted = bool(stream.exhausted)
            except Exception:
                exhausted = True
            return {"enabled": bool(enabled), "open": opened,
                    "exhausted": exhausted}

        payload = {
            "ts": now_wall,
            "app": "run_workstate",
            "single_channel": False,
            "uptime_s": now_s,
            "camera_a": _stream_state(stream_a, True),
            "camera_b": _stream_state(stream_b, True),
            "count": int(count_a.value + count_b.value),
            "count_a": int(count_a.value),
            "count_b": int(count_b.value),
            "people": people,
            "inference_ms": infer_ms,
            "loop_fps": fps,
            "attendance_today": att,
            "recent_events": events,
            "models": {
                "yolo": str(model_path),
                "reid": str(getattr(config.identity, "reid_model", "?")),
                "face": ("bat" if shared_face_embedder is not None else "tat"),
            },
        }
        try:
            target = _PROJECT_ROOT / "output" / "qa_cache" / "runtime_status.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_text(_json.dumps(payload, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(target)
        except OSError:
            pass

    try:
        while True:
            now_s = time.time() - start
            local_day = datetime.now().astimezone().date().isoformat()
            if local_day != current_identity_day:
                identity_store.save(current_identity_day, manager)
                old_gids = set(manager.identities)
                for gid in old_gids:
                    fusion.forget(gid)
                    business_a.forget(gid)
                    business_b.forget(gid)
                manager.reset_for_new_day()
                identity_store.load(local_day, manager)
                gid_to_person.clear()
                gid_to_display.clear()
                gid_to_score.clear()
                gid_to_face_candidates.clear()
                gid_to_face_rankings.clear()
                gid_face_ranking_at.clear()
                face_assignment_streaks.clear()
                face_candidate_streaks.clear()
                face_quality_by_track.clear()
                recent_face_by_track.clear()
                gid_alias.clear()
                unknown_of_gid.clear()
                unknown_counter = 0
                for _selector in foreground_selectors.values():
                    _selector.reset()
                foreground_by_channel.update({"A": None, "B": None})
                if face_worker is not None:
                    face_worker.forget_retired(set())
                for _det in list(gesture_detectors.values()):
                    try:
                        _det.forget_retired(set())
                    except AttributeError:
                        pass
                if palm_detector is not None:
                    try:
                        palm_detector.forget_retired(set())
                    except AttributeError:
                        pass
                current_identity_day = local_day
            ret_a, frame_a = stream_a.read()
            ret_b, frame_b = stream_b.read()
            if not ret_a and not ret_b:
                if stream_a.exhausted and stream_b.exhausted:
                    print("Hết stream cả 2 camera.")
                    break
                time.sleep(0.05)
                continue
            process_frame = frame_idx % config.camera.process_every_n_frames == 0

            if ret_a and frame_a is not None:
                frame_a = normalize_frame_size(
                    frame_a, config.camera.frame_width, config.camera.frame_height
                )
            if ret_b and frame_b is not None:
                frame_b = normalize_frame_size(
                    frame_b, config.camera.frame_width, config.camera.frame_height
                )
            if ret_a and frame_a is not None:
                frame_hub.publish("A", Frame(frame_idx, now_s, frame_a))
            if ret_b and frame_b is not None:
                frame_hub.publish("B", Frame(frame_idx, now_s, frame_b))

            batch_detections: dict[str, list] = {}
            if process_frame:
                face_marks_a.clear()
                face_marks_b.clear()
                # ALWAYS-ON: YOLO batch 2 camera (GPU). Face KHÔNG chạy ở đây
                # nữa — event-driven once-per-track qua FaceWorker (async).
                batch_names: list[str] = []
                batch_images: list[np.ndarray] = []
                if ret_a and frame_a is not None:
                    batch_names.append("A")
                    batch_images.append(frame_a)
                if ret_b and frame_b is not None:
                    batch_names.append("B")
                    batch_images.append(frame_b)
                with metrics.measure("detection"):
                    batch_detections = dict(
                        zip(batch_names, detector.detect_batch(batch_images), strict=True)
                    )

            # Local trackers run independently; global association sees both
            # cameras at one timestamp and may return the same GID in overlap.
            if process_frame:
                identity_inputs: dict[str, tuple[np.ndarray, list[Track]]] = {}
                for channel, frame, tracker in (
                    ("A", frame_a if ret_a else None, tracker_a),
                    ("B", frame_b if ret_b else None, tracker_b),
                ):
                    if frame is None:
                        continue
                    with metrics.measure(f"tracking_{channel}"):
                        local_tracks = tracker.update(batch_detections[channel])
                    local_tracks = [
                        track for track in local_tracks
                        if track.confirmed
                        and track.bbox.area >= args.min_area
                        and _sane_box(
                            track.bbox,
                            config.camera.frame_width,
                            config.camera.frame_height,
                        )
                    ]
                    identity_inputs[channel] = (frame, local_tracks)
                global_tracks = manager.update_batch(identity_inputs, now_s=now_s)
                confirmed_a = global_tracks.get("A", [])
                confirmed_b = global_tracks.get("B", [])
                foreground_by_channel["A"] = foreground_selectors["A"].update(
                    confirmed_a, now_s)
                foreground_by_channel["B"] = foreground_selectors["B"].update(
                    confirmed_b, now_s)
                if ret_a and frame_a is not None:
                    last_tracks_a = confirmed_a
                    count_a.update(len(confirmed_a))
                    floor_a = (
                        _track_floor_points(confirmed_a, projector)
                        if use_roi
                        else _track_norm_centers(
                            confirmed_a,
                            config.camera.frame_width,
                            config.camera.frame_height,
                        )
                    )
                    states_a = workstate_a.consume(
                        TrackEvent("A", Frame(frame_idx, now_s, frame_a), tuple(confirmed_a)),
                        floor_a,
                        gid_to_person,
                    )
                    last_states_a = {gid: state.value for gid, state in states_a.items()}
                if ret_b and frame_b is not None:
                    last_tracks_b = confirmed_b
                    count_b.update(len(confirmed_b))
                    floor_b = (
                        _track_floor_points(confirmed_b, projector)
                        if use_roi
                        else _track_norm_centers(
                            confirmed_b,
                            config.camera.frame_width,
                            config.camera.frame_height,
                        )
                    )
                    try:
                        states_b = workstate_b.consume(
                            TrackEvent("B", Frame(frame_idx, now_s, frame_b), tuple(confirmed_b)),
                            floor_b,
                            gid_to_person,
                        )
                    except (KeyError, ValueError, RuntimeError) as error:
                        print(f"Workstate B skipped for frame: {error}")
                        states_b = {}
                    last_states_b = {gid: state.value for gid, state in states_b.items()}
                # EVENT-DRIVEN Face: enqueue once-per-track (rẻ, không block),
                # worker GPU xử lý nền, drain kết quả mỗi frame.
                now_dt = datetime.now().astimezone()
                _day = now_dt.date().isoformat()
                _wall = now_dt.isoformat(timespec="seconds")
                _tag = now_dt.strftime("%H%M%S")
                if face_worker is not None:
                    for _ch, _fr, _tr in (
                        ("A", frame_a if ret_a else None, confirmed_a),
                        ("B", frame_b if ret_b else None, confirmed_b),
                    ):
                        if _fr is not None and _tr:
                            with metrics.measure("face_enqueue"):
                                _enqueue_face_async(
                                    channel=_ch, frame=_fr,
                                    confirmed_tracks=_tr,
                                    day_str=_day, wall_iso=_wall,
                                    time_tag=_tag, now_s=now_s,
                                )
                    _drain_face_results(
                        frame_a=frame_a if ret_a else None,
                        frame_b=frame_b if ret_b else None,
                        face_marks_a=face_marks_a,
                        face_marks_b=face_marks_b,
                    )
                else:
                    # Fallback sync (không worker): giữ cadence cũ xen kẽ A/B.
                    face_slot = (frame_idx // max(1, config.camera.process_every_n_frames)) \
                        // max(1, face_cfg.process_every_k)
                    face_order = [c for c in ("A", "B") if c in face_cfg.channels]
                    face_pick = face_order[face_slot % len(face_order)] if face_order else None
                    for channel, frame, tracks, marks in (
                        ("A", frame_a if ret_a else None, confirmed_a, face_marks_a),
                        ("B", frame_b if ret_b else None, confirmed_b, face_marks_b),
                    ):
                        if frame is not None and use_face and face_pick == channel:
                            _process_face_channel(
                                channel=channel, frame=frame,
                                confirmed_tracks=tracks, face_marks=marks,
                                day_str=_day, wall_iso=_wall,
                                time_tag=_tag, now_s=now_s,
                            )

            active_merges = manager.reconcile_active_duplicates()
            for duplicate_gid, canonical_gid in active_merges.items():
                gid_alias[duplicate_gid] = canonical_gid
                # Business state is keyed by Global ID too.  Move that state
                # before rendering, otherwise the canonical person can still
                # appear AWAY while the duplicate is visible/WORKING.
                for tracker in (business_a, business_b):
                    try:
                        tracker.transfer_assignment(duplicate_gid, canonical_gid)
                    except (KeyError, ValueError) as error:
                        print(
                            f"Workstate merge G{duplicate_gid}->{canonical_gid} "
                            f"skipped: {error}"
                        )
                print(
                    f"[Reconcile] active G{duplicate_gid} -> "
                    f"G{canonical_gid} (high ReID similarity)"
                )

            # Face confirmation may merge an active duplicate GID into the
            # older canonical GID. Rewrite live tracks before fusion/render.
            last_tracks_a = _remap_tracks(last_tracks_a, gid_alias, manager)
            last_tracks_b = _remap_tracks(last_tracks_b, gid_alias, manager)
            # Move transient face/gesture state to the canonical identity so
            # an ID reconciliation cannot break focus halfway through a wave.
            for _old_gid, _canonical_gid in list(gid_alias.items()):
                for _cache, _fresh_index in (
                    (recent_face_by_track, 0), (face_quality_by_track, 1)
                ):
                    for _channel in ("A", "B"):
                        _old_key = (_channel, _old_gid)
                        _new_key = (_channel, _canonical_gid)
                        if _old_key in _cache:
                            _old_value = _cache.pop(_old_key)
                            _current = _cache.get(_new_key)
                            if (_current is None
                                    or _old_value[_fresh_index]
                                    >= _current[_fresh_index]):
                                _cache[_new_key] = _old_value
                for _gesture_detector in list(gesture_detectors.values()):
                    try:
                        _gesture_detector.remap_gid(_old_gid, _canonical_gid)
                    except AttributeError:
                        pass
            for _channel, _tracks in (("A", last_tracks_a), ("B", last_tracks_b)):
                foreground_selectors[_channel].remap(gid_alias)
                foreground_by_channel[_channel] = foreground_selectors[_channel].update(
                    _tracks, now_s)
            if process_frame:
                identity_trace.write(frame_idx, "A", last_tracks_a, gid_to_person)
                identity_trace.write(frame_idx, "B", last_tracks_b, gid_to_person)

            # --- Fuse trang thai phong A+B (moi frame xu ly) ---
            if process_frame:
                now_dt = datetime.now().astimezone()
                day_str = now_dt.date().isoformat()
                wall_iso = now_dt.isoformat(timespec="seconds")
                present_a = {t.track_id for t in last_tracks_a}
                present_b = {t.track_id for t in last_tracks_b}
                _snapshot_new_gids(
                    seen_gids, last_tracks_a, last_tracks_b,
                    frame_a if ret_a else None, frame_b if ret_b else None,
                    ghosts_dir,
                )
                room_status_now = fusion.update(
                    now_s, present_a=present_a, present_b=present_b,
                    states_a=last_states_a, states_b=last_states_b,
                    names=gid_to_display,
                )
                for gid, st in room_status_now.items():
                    canon = gid_alias.get(gid)
                    if canon is not None and canon != gid and canon in room_status_now:
                        continue  # canonical gid carries the live status
                    wgid = canon if canon is not None else gid
                    # Tentative identities (too young) never touch the DB:
                    # ghost boxes die here instead of polluting history.
                    record = manager.identities.get(wgid)
                    established = record is not None
                    write_row, log_event = day_cache.status_should_write(
                        day=day_str, global_id=wgid, label=st.label,
                        in_room=st.in_room, now_s=now_s,
                    )
                    if st.just_left_office:
                        day_cache.note_leave_enter(day_str, wgid, leave_at=wall_iso)
                    if st.just_returned:
                        day_cache.note_leave_enter(day_str, wgid, enter_at=wall_iso)
                    if log_event and established and (
                            st.just_left_office or st.just_returned):
                        _store_or_queue(write_queue, supabase, "event", {
                            "date": day_str, "global_id": wgid,
                            "person_id": gid_to_person.get(wgid),
                            "event": "LEAVE_OFFICE" if st.just_left_office else "RETURN",
                            "channel": "B" if st.just_left_office else "A",
                            "at": wall_iso,
                        })
                        _note_event(
                            event="LEAVE_OFFICE" if st.just_left_office else "RETURN",
                            global_id=wgid,
                            channel="B" if st.just_left_office else "A",
                            at_iso=wall_iso,
                            person_id=gid_to_person.get(wgid),
                            person_name=gid_to_display.get(wgid),
                        )
                    if write_row and established:
                        row = day_cache.status_of(day_str, wgid)
                        if row is None:
                            continue
                        _store_or_queue(write_queue, supabase, "room_status", {
                            "date": day_str, "global_id": wgid,
                            "person_id": gid_to_person.get(wgid),
                            "person_name": gid_to_display.get(wgid),
                            "in_room": st.in_room, "label": st.label,
                            "last_leave_at": row.last_leave_at,
                            "last_enter_at": row.last_enter_at,
                        })
                        employee_id = gid_to_person.get(wgid)
                        if employee_id:
                            _store_or_queue(write_queue, supabase, "current_state", {
                                "employee_id": employee_id,
                                "state": st.label,
                                "since": row.last_enter_at or wall_iso,
                                "camera_id": "A" if gid in present_a else "B",
                                "global_id": wgid,
                                "confidence": gid_to_score.get(wgid),
                                "updated_at": wall_iso,
                            })
                if now_s - last_flush_s >= config.room_fusion.flush_s:
                    last_flush_s = now_s
                    _prune_identity_maps(
                        manager, gid_to_person, gid_to_display,
                        gid_to_score, unknown_of_gid, reconciler,
                        fusion, (business_a, business_b),
                        face_consumer, attendance, now_s,
                    )
                    try:
                        _alive = set(manager.identities)
                        if face_worker is not None:
                            face_worker.forget_retired(_alive)
                        for _det in list(gesture_detectors.values()):
                            try:
                                _det.forget_retired(_alive)
                            except AttributeError:
                                pass
                        if palm_detector is not None:
                            try:
                                palm_detector.forget_retired(_alive)
                            except AttributeError:
                                pass
                        for gid in [g for g in gid_to_face_candidates if g not in _alive]:
                            gid_to_face_candidates.pop(gid, None)
                        for gid in [g for g in gid_to_face_rankings if g not in _alive]:
                            gid_to_face_rankings.pop(gid, None)
                            gid_face_ranking_at.pop(gid, None)
                            face_assignment_streaks.pop(gid, None)
                        for gid in [g for g in face_candidate_streaks if g not in _alive]:
                            face_candidate_streaks.pop(gid, None)
                        for key in [k for k in recent_face_by_track if k[1] not in _alive]:
                            recent_face_by_track.pop(key, None)
                        for key in [k for k in face_quality_by_track if k[1] not in _alive]:
                            face_quality_by_track.pop(key, None)
                    except Exception:
                        pass
                    identity_store.save(day_str, manager)
                # --- Chao khach la tre: den han ma van chua dinh danh thi
                # moi chao "quy khach"; dinh danh kip thi nhanh known da huy
                # hen (chi chao ten). Chay moi frame, re nhu dict nho.
                if voice_on and greeter is not None and pending_unknown_greet:
                    try:
                        for _ug_gid in list(pending_unknown_greet):
                            _due, _day, _ch, _iso, _spk = \
                                pending_unknown_greet[_ug_gid]
                            if (
                                _day != day_str
                                or _ug_gid not in manager.identities
                                or gid_to_person.get(_ug_gid) is not None
                                or _ug_gid in gid_alias
                                or foreground_by_channel.get(_ch) != _ug_gid
                            ):
                                pending_unknown_greet.pop(_ug_gid, None)
                                continue
                            if now_s < _due:
                                continue
                            pending_unknown_greet.pop(_ug_gid, None)
                            if (not _ha_turn_active() and greeter.face_greet(
                                day=_day, person_id=None,
                                display_name=None,
                                now_s=now_s, global_id=_ug_gid,
                                channel=_spk,
                            )):
                                _note_event(
                                    event="FACE_GREET", global_id=_ug_gid,
                                    channel=_ch, at_iso=_iso,
                                )
                    except Exception as error:  # noqa: BLE001
                        print(f"[Voice] Bo loi chao khach tre: {error}")
                # --- Gesture theo kenh -> greet (vay tay/palm only, da bo voice-hello) ---
                if voice_on and greeter is not None and gesture_detectors:
                    voice_tick += 1
                    _has_wave = any(
                        m == "wave" for m in gesture_modes.values()
                    )
                    if _has_wave:
                        _gesture_every = max(1, int(getattr(
                            voice_cfg, "wave_every_k", 3)))
                    else:
                        _gesture_every = max(1, int(getattr(
                            voice_cfg, "palm_every_k",
                            getattr(voice_cfg, "wave_every_k", 4))))
                    if voice_tick % _gesture_every == 0:
                        with metrics.measure("wave"):
                            for _ch, _frame, _tracks in (
                                ("A", frame_a if ret_a else None, last_tracks_a),
                                ("B", frame_b if ret_b else None, last_tracks_b),
                            ):
                                if _frame is None:
                                    continue
                                _mode = gesture_modes.get(_ch, "off")
                                _det = gesture_detectors.get(_ch)
                                if _det is None:
                                    continue
                                # Gesture and voice target only the stable
                                # foreground subject (largest bbox + hysteresis).
                                # Background people remain face-attendance only.
                                _focus_gid = foreground_by_channel.get(_ch)
                                _focus_track = next(
                                    (t for t in _tracks
                                     if t.track_id == _focus_gid), None)
                                if _focus_track is None:
                                    continue
                                for _track in (_focus_track,):
                                    _gid = _track.track_id
                                    # Gate: bbox quá nhỏ thì tay không đủ pixel.
                                    if (_det is not None
                                            and _track.bbox.area < float(getattr(
                                                voice_cfg, "palm_min_person_area_px", 8000.0))):
                                        continue
                                    # Quen: chi chao ten khi face da bind (chong chao nham).
                                    _bound_employee = manager.employee_id_of(_gid)
                                    _mapped_employee = gid_to_person.get(_gid)
                                    if (_bound_employee is not None
                                            and _mapped_employee == _bound_employee):
                                        _pid = _bound_employee
                                        _name = gid_to_display.get(_gid)
                                        if not _name:
                                            continue
                                    elif _gid in unknown_of_gid:
                                        _pid = _name = None
                                    elif getattr(voice_cfg, "palm_unknown_immediate", True):
                                        _pid = _name = None
                                    else:
                                        if now_s - wave_hint_at.get(_gid, float("-inf")) > 5.0:
                                            wave_hint_at[_gid] = now_s
                                            print(f"[Gesture][{_ch}] G{_gid} thay tay "
                                                  f"nhung chua nhan dien mat")
                                        continue
                                    # Loa phat theo kenh camera (A->ch1, B->ch2).
                                    # Gesture/voice la yeu cau chu dong, nen khong
                                    # bi cooldown cua face-greet tu dong chan.
                                    # VoiceGreeter tu gop queue va ap quiet-window
                                    # 5s/nguoi de chong spam sau khi phat.
                                    _spk = _speaker_channel(_ch)
                                    _gesture_label = (
                                        "Wave" if _mode == "wave"
                                        else "Palm" if _mode == "palm" else "Voice"
                                    )
                                    _crop = _person_crop(_frame, _track.bbox)
                                    if _crop is None:
                                        continue
                                    # Nếu vừa thấy mặt thì dùng vị trí mặt để
                                    # kiểm tra tay cùng người. Không ép phải có
                                    # face mới: người nghiêng mặt vẫn có thể chủ
                                    # động xòe đủ 5 ngón và vẫy để chào.
                                    _face_state = recent_face_by_track.get((_ch, _gid))
                                    _face_ttl = float(getattr(
                                        voice_cfg, "wave_face_ttl_s",
                                        getattr(voice_cfg, "palm_face_ttl_s", 3.0)))
                                    _fresh_face = bool(
                                        _face_state is not None
                                        and now_s - _face_state[0] <= _face_ttl
                                    )
                                    _face_box_px = None
                                    if _fresh_face and _face_state is not None:
                                        _nh = _crop.shape[0]
                                        _nw = _crop.shape[1]
                                        _nb = _face_state[1]
                                        _face_box_px = (
                                            _nb[0] * _nw, _nb[1] * _nh,
                                            _nb[2] * _nw, _nb[3] * _nh,
                                        )
                                    gestured = False
                                    if _det is not None and _mode == "wave":
                                        try:
                                            gestured = _det.observe_open_palm_wave(
                                                _track.track_id, _crop, now_s,
                                                face_bbox=_face_box_px,
                                                face_max_distance=float(getattr(
                                                    voice_cfg,
                                                    "wave_face_max_distance", 3.5)),
                                            )
                                        except RuntimeError:
                                            gestured = False
                                        except (AttributeError, TypeError, ValueError):
                                            gestured = False
                                    elif _det is not None and _mode == "palm":
                                        _allow_no_face = bool(getattr(
                                            voice_cfg, "palm_allow_without_face", False))
                                        if ((_fresh_face and _face_state is not None)
                                                or _allow_no_face):
                                            try:
                                                gestured = _det.observe(
                                                    _track.track_id, _crop, now_s,
                                                    face_bbox=(_face_state[1]
                                                               if _fresh_face and _face_state
                                                               else None),
                                                    face_max_distance=float(getattr(
                                                        voice_cfg,
                                                        "palm_face_max_distance", 2.5)),
                                                )
                                            except RuntimeError:
                                                gestured = False
                                    # Chi tay (wave/palm): hello imou da bo.
                                    if not gestured:
                                        # Log chan doan throttle theo GID.
                                        if now_s - palm_dbg_at.get(
                                                _gid, float("-inf")) > 15.0:
                                            palm_dbg_at[_gid] = now_s
                                            if _mode == "wave":
                                                try:
                                                    _rev = _det.current_reversals(_gid)
                                                except (AttributeError, TypeError, ValueError):
                                                    _rev = -1
                                                print(f"[Wave][{_ch}] G{_gid} chua du "
                                                      f"dao chieu ({_rev})")
                                            else:
                                                _reason = ("mat va ban tay cung luc"
                                                           if not _fresh_face
                                                           else "ban tay 4-5 ngon gan mat")
                                                print(f"[Palm][{_ch}] G{_gid} khong thay "
                                                      f"{_reason}")
                                        continue
                                    # Yeu cau chu dong (vay tay): phat ngay khi
                                    # loa ranh, khong cooldown, chong spam 5s.
                                    # Nhuong turn Ha Linh: dang hoi-dap thi bo
                                    # qua, tranh 2 loa noi chong nhau.
                                    if (not _ha_turn_active()
                                            and greeter.face_greet(
                                                day=day_str, person_id=_pid,
                                                display_name=_name, now_s=now_s,
                                                global_id=_gid, channel=_spk,
                                                proactive=True,
                                            )):
                                        _note_event(
                                            event=("WAVE" if _mode == "wave" else "PALM"),
                                            global_id=_gid,
                                            channel=_ch, at_iso=wall_iso,
                                            person_id=_pid, person_name=_name,
                                        )
                                        _tag = ("Wave" if _mode == "wave" else "Palm")
                                        _why = (("vẫy tay" if _mode == "wave"
                                                      else "giơ tay"))
                                        print(f"[{_tag}][{_ch}] "
                                              f"{_name or 'Unknown'} (G{_gid}, {_why}, "
                                              f"loa ch{_spk})")
                                    elif now_s - palm_dbg_at.get(_gid, float("-inf")) > 5.0:
                                        # Hiem (don thread): dat cho that bai do race
                                        # voi worker hoan reservation.
                                        palm_dbg_at[_gid] = now_s
                                        if _ha_turn_active():
                                            print(f"[{_gesture_label}][{_ch}] G{_gid} dang "
                                                  f"hoi-dap Ha Linh, bo qua chao vẫy")
                                        else:
                                            print(f"[{_gesture_label}][{_ch}] G{_gid} loa vua ban, "
                                                  f"vẫy lại sau ít giây")

            # --- Live view: annotated Global-ID frames for window/stream ---
            need_vis = args.display or stream_on
            vis_a = vis_b = None
            if ret_a and frame_a is not None and need_vis:
                vis_a = frame_a.copy()
                draw_person_tracks(vis_a, last_tracks_a, display_count=count_a.value,
                                   title="Global", status=room_status_now)
                draw_global_labels(
                    vis_a, last_tracks_a, room_status_now, gid_to_display,
                    gid_to_person, gid_to_face_candidates
                )
                for x1m, y1m, x2m, y2m, fscore, fknown, fgid in face_marks_a:
                    fcolor = (40, 180, 40) if fknown else (60, 60, 220)
                    cv2.rectangle(vis_a, (int(x1m), int(y1m)),
                                  (int(x2m), int(y2m)), fcolor, 2)
                    cv2.putText(vis_a, f"{fscore:.2f}",
                                (int(x1m), max(0, int(y1m) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fcolor, 2)
                _wave_a = gesture_detectors.get("A")
                _focus_a = foreground_by_channel.get("A")
                _focus_track_a = next(
                    (t for t in last_tracks_a if t.track_id == _focus_a), None)
                if _wave_a is not None and _focus_track_a is not None:
                    try:
                        _hand_debug = _wave_a.debug_snapshot(_focus_a, now_s)
                    except (AttributeError, TypeError, ValueError):
                        _hand_debug = None
                    if _hand_debug is not None:
                        draw_hand_landmarks(
                            vis_a, _focus_track_a.bbox,
                            _hand_debug[0], _hand_debug[1],
                            reversals=_wave_a.current_reversals(_focus_a),
                            required_reversals=int(getattr(
                                voice_cfg, "wave_min_reversals", 2)),
                        )
            if ret_b and frame_b is not None and need_vis:
                vis_b = frame_b.copy()
                draw_person_tracks(vis_b, last_tracks_b, display_count=count_b.value,
                                   title="Global", status=room_status_now)
                draw_global_labels(
                    vis_b, last_tracks_b, room_status_now, gid_to_display,
                    gid_to_person, gid_to_face_candidates
                )
                for x1m, y1m, x2m, y2m, fscore, fknown, fgid in face_marks_b:
                    fcolor = (40, 180, 40) if fknown else (60, 60, 220)
                    cv2.rectangle(vis_b, (int(x1m), int(y1m)),
                                  (int(x2m), int(y2m)), fcolor, 2)
                    cv2.putText(vis_b, f"{fscore:.2f}",
                                (int(x1m), max(0, int(y1m) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fcolor, 2)
                _wave_b = gesture_detectors.get("B")
                _focus_b = foreground_by_channel.get("B")
                _focus_track_b = next(
                    (t for t in last_tracks_b if t.track_id == _focus_b), None)
                if _wave_b is not None and _focus_track_b is not None:
                    try:
                        _hand_debug = _wave_b.debug_snapshot(_focus_b, now_s)
                    except (AttributeError, TypeError, ValueError):
                        _hand_debug = None
                    if _hand_debug is not None:
                        draw_hand_landmarks(
                            vis_b, _focus_track_b.bbox,
                            _hand_debug[0], _hand_debug[1],
                            reversals=_wave_b.current_reversals(_focus_b),
                            required_reversals=int(getattr(
                                voice_cfg, "wave_min_reversals", 2)),
                        )
            if args.display:
                if vis_a is not None:
                    cv2.imshow("Channel A - tracking", vis_a)
                    window_a_shown = True
                if vis_b is not None:
                    cv2.imshow("Channel B - tracking", vis_b)
                    window_b_shown = True
            if stream_on:
                with metrics.measure("rendering"):
                    if jpeg_renderer is not None:
                        if vis_a is not None:
                            jpeg_renderer.submit("cam_a", vis_a)
                        if vis_b is not None:
                            jpeg_renderer.submit("cam_b", vis_b)
                if process_frame:
                    live_people = []
                    for gid, st in sorted(room_status_now.items()):
                        identity = manager.identities.get(gid)
                        live_people.append({
                            "gid": gid,
                            "person_id": gid_to_person.get(gid),
                            "name": st.display_name,
                            "label": st.label,
                            "in_room": st.in_room,
                            "face_score": gid_to_score.get(gid),
                            "identity_state": (
                                "employee" if gid_to_person.get(gid) else "unknown"
                            ),
                            "identity_confidence": (
                                identity.last_match_score if identity else None
                            ),
                            "employee_confidence": gid_to_score.get(gid),
                            "face_candidates": [
                                {"name": name, "score": score}
                                for name, score in gid_to_face_candidates.get(gid, [])[:2]
                            ],
                            "camera": identity.channel if identity else None,
                            "cameras": (
                                sorted(identity.sightings) if identity else []
                            ),
                            "tracking_state": (
                                identity.state.value if identity else None
                            ),
                            "foreground_cameras": [
                                channel for channel in ("A", "B")
                                if foreground_by_channel.get(channel) == gid
                            ],
                        })
                    attendance_today = [] if attendance is None else [
                        {
                            "date": record.day,
                            "person_id": record.person_id,
                            "person_name": record.display_name,
                            "global_id": record.global_id,
                            "attended": True,
                            "check_in_at": record.wall_time,
                            "face_score": record.face_score,
                        }
                        for record in attendance.records_for_day(day_str)
                    ]
                    streamer.set_status({
                        "people": live_people,
                        "attendance_today": attendance_today,
                        "recent_events": list(recent_events),
                        "pending_attendance": pending_attendance(),
                        "count_a": count_a.value,
                        "count_b": count_b.value,
                    })

            # Cau runtime cho voice Ha Linh BAN CAMERA (2s/lan): dem nguoi,
            # camera, fps, diem danh, su kien -> tool tra loi so THAT.
            if process_frame and now_s - last_runtime_write_s >= 2.0:
                last_runtime_write_s = now_s
                try:
                    write_runtime_status()
                except Exception:
                    pass
                try:
                    poll_voice_command()
                except Exception:
                    pass
            if args.display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("Đã nhấn phím thoát.")
                    break
                window_a_closed = window_a_shown and cv2.getWindowProperty(
                    "Channel A - tracking", cv2.WND_PROP_VISIBLE
                ) < 1
                window_b_closed = window_b_shown and cv2.getWindowProperty(
                    "Channel B - tracking", cv2.WND_PROP_VISIBLE
                ) < 1
                if window_a_closed or window_b_closed:
                    print("Cửa sổ đã đóng. Đang giải phóng camera...")
                    break
            if args.metrics_log_s > 0 and now_s - last_metrics_log_s >= args.metrics_log_s:
                last_metrics_log_s = now_s
                print(f"[Metrics] {metrics.snapshot()}")
            frame_idx += 1
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
    finally:
        identity_trace.close()
        identity_store.save(
            current_identity_day, manager
        )
        try:
            if face_worker is not None:
                face_worker.stop()
        except Exception as error:  # noqa: BLE001
            print(f"Face worker stop loi: {error}")
        try:
            write_worker.stop()
        except Exception as error:  # noqa: BLE001
            print(f"Flush cuoi loi (da giu trong queue): {error}")
        if jpeg_renderer is not None:
            jpeg_renderer.stop()
        if streamer is not None:
            streamer.stop()
        if supervisor_server is not None:
            try:
                supervisor_server.shutdown()
                supervisor_server.server_close()
            except Exception as error:  # noqa: BLE001
                print(f"Supervisor stop loi: {error}")
        if voice_bridge is not None:
            voice_bridge.close()
        try:
            if voice_stop is not None:
                voice_stop.set()
        except Exception as error:  # noqa: BLE001
            print(f"VoiceTrigger stop loi: {error}")
        if halinh_stop is not None:
            halinh_stop.set()
        if halinh_thread is not None:
            halinh_thread.join(timeout=15.0)
        stream_a.close()
        stream_b.close()
        cv2.destroyAllWindows()
    elapsed = time.time() - start
    gids = sorted(manager.identities)
    print(f"Xong sau {elapsed:.1f}s. Count A: {count_a.value} | Count B: {count_b.value}")
    print(f"Stage metrics: {metrics.snapshot()}")
    print(f"Global IDs: {gids}")
    for gid in gids:
        record = manager.identities[gid]
        print(f"  G{gid}: state={record.state.value} last_channel={record.channel} "
              f"hits={record.total_hits} gallery={len(record.gallery)}")
    ghost_hits = max(10, 2 * identity_cfg.tentative_min_hits)
    ghosts = sorted(
        ((gid, manager.identities[gid].total_hits) for gid in gids
         if manager.identities[gid].total_hits < ghost_hits),
        key=lambda item: item[1],
    )
    if ghosts:
        print(f"Nghi ghost ({len(ghosts)}, <{ghost_hits} hits): "
              + ", ".join(f"G{g}({h})" for g, h in ghosts)
              + f" — crop dau xem tai {ghosts_dir}/")
    if attendance is not None:
        day_now = datetime.now().astimezone().date().isoformat()
        ticked = attendance.records_for_day(day_now)
        print(f"Diem danh {day_now}: {len(ticked)} nguoi")
        for rec in ticked:
            print(f"  {rec.display_name} (G{rec.global_id}, "
                  f"score={rec.face_score:.2f}, in={rec.wall_time})")
        if unknown_of_gid:
            print(f"Mat la: {len(unknown_of_gid)} ({', '.join(sorted(unknown_of_gid.values()))})")
    pending = len(write_queue)
    if pending:
        print(f"Queue ton {pending} writes (offline/miss env) -> se flush lan chay sau.")


if __name__ == "__main__":
    main()
