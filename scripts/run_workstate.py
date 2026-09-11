"""Demo tracking Global ID 2 channel IMOU.

Moi channel chay doc lap: YOLO(26s) -> ByteTrack (tracking ngan han:
motion prediction, IoU matching, data association, track buffer).
Mot GlobalIdentityManager dung chung cho ca 2 channel tra loi
"day la nguoi nao" (Global Person ID on dinh xuyen tracklet, xuyen mat dau
dai, xuyen channel) bang appearance gallery + cost matrix + Hungarian +
gating + lifecycle ACTIVE/TEMP_LOST/LONG_LOST/UNRESOLVED.
Lop business theo channel (ChannelBusinessTracker) tra loi rieng
"nguoi do dang lam gi" (WORKING/AWAY_TEMP/POSSIBLY_OUT/RETURNING) va khong
duoc phep anh huong toi ID.

Chay:
  python scripts/run_workstate.py --config config/default.yaml --display
  python scripts/run_workstate.py --source-a 0 --source-b data/samples/hallway.mp4 --display

RTSP IMOU lay tu .env (IMOU_IP/USER/PASSWORD) neu khong truyen --source.
Nhan 'q' hoac ESC de thoat.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np

from camera_tracking.config import load_config
from camera_tracking.detection import YoloPersonDetector, resolve_device
from camera_tracking.tracking import (
    ByteTrackTracker,
    GlobalIdentityConfig,
    GlobalIdentityManager,
    StablePersonCount,
)
from camera_tracking.visualization import draw_person_tracks
from camera_tracking.workstate import (
    ChannelBusinessTracker,
    HistogramEmbedding,
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
        5000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        5000,
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
    parser = argparse.ArgumentParser(description="Tracking ID 2 camera (YOLO + ByteTrack + Re-ID).")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--source-a", default=None, help="Override camera A.")
    parser.add_argument("--source-b", default=None, help="Override camera B.")
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
        "--subtype-b", type=int, default=1, choices=(0, 1), help="0=main, 1=sub stream."
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
    parser.add_argument("--match-threshold", type=float, default=None,
                        help="Override identity match_threshold trong config.")
    parser.add_argument("--away-grace-s", type=float, default=10.0,
                        help="Business: vang qua N giay -> AWAY_TEMP.")
    parser.add_argument("--out-after-s", type=float, default=60.0,
                        help="Business: vang qua N giay -> POSSIBLY_OUT.")
    parser.add_argument("--return-stable-s", type=float, default=3.0,
                        help="Business: hien dien on dinh N giay -> WORKING.")
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu / auto (mac dinh lay theo config).")
    return parser.parse_args()


def normalize_frame_size(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize moi source ve kich thuoc chung de hien thi on dinh."""
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    interpolation = cv2.INTER_AREA if frame.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def draw_business_states(frame: np.ndarray, tracks, states: dict[int, str]) -> np.ndarray:
    """Ve them business state ke ben box cua tung Global ID (chi de hien thi)."""
    for track in tracks:
        label = f"G{track.track_id}:{states.get(track.track_id, '?')}"
        x1, y1 = max(0, int(track.bbox.x1)), max(0, int(track.bbox.y1))
        cv2.putText(frame, label, (x1, max(20, y1 - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def main() -> None:
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
    print(
        f"Device: {device} | Model: {model_path} @ {image_size}px | "
        f"Camera A: {source_label(source_a)} | "
        f"Camera B: {source_label(source_b)}"
    )

    detector = YoloPersonDetector(
        model_path=model_path,
        # ByteTrack uses weak detections to recover an existing person through occlusion.
        confidence=min(0.10, config.detection.confidence_threshold),
        person_class_id=config.detection.person_class_id,
        image_size=image_size,
        device=device,
    )
    effective_fps = max(1, round(config.camera.fps / config.camera.process_every_n_frames))
    tracker_a = ByteTrackTracker(
        frame_rate=effective_fps,
        track_buffer=30,
        match_threshold=0.9,
        min_hits=config.tracking.min_hits,
    )
    tracker_b = ByteTrackTracker(
        frame_rate=effective_fps,
        track_buffer=30,
        match_threshold=0.9,
        min_hits=config.tracking.min_hits,
    )
    embedder = HistogramEmbedding()
    identity_cfg = config.identity
    if args.match_threshold is not None:
        identity_cfg = identity_cfg.model_copy(
            update={"match_threshold": args.match_threshold}
        )
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
            min_appearance_similarity=identity_cfg.min_appearance_similarity,
            temp_lost_s=identity_cfg.temp_lost_s,
            long_lost_s=identity_cfg.long_lost_s,
            unresolved_keep_s=identity_cfg.unresolved_keep_s,
            min_gallery_confidence=identity_cfg.min_gallery_confidence,
        ),
    )
    # Business theo channel: chi doc Global ID + presence, khong anh huong ID.
    business_a = ChannelBusinessTracker(
        channel="A",
        away_grace_s=args.away_grace_s,
        out_after_s=args.out_after_s,
        return_stable_s=args.return_stable_s,
    )
    business_b = ChannelBusinessTracker(
        channel="B",
        away_grace_s=args.away_grace_s,
        out_after_s=args.out_after_s,
        return_stable_s=args.return_stable_s,
    )
    count_a = StablePersonCount(rise_frames=3, fall_frames=12)
    count_b = StablePersonCount(rise_frames=3, fall_frames=12)

    cap_a = open_capture(source_a, attempts=args.open_attempts, camera_name="A")
    cap_b = open_capture(source_b, attempts=args.open_attempts, camera_name="B")
    if not cap_a.isOpened():
        print(f"Không mở được camera A: {source_label(source_a)}")
    if not cap_b.isOpened():
        print(f"Không mở được camera B: {source_label(source_b)}")
    if not cap_a.isOpened() and not cap_b.isOpened():
        raise SystemExit(1)
    stream_a = ResilientCapture(
        source_a,
        "A",
        cap_a,
        args.open_attempts,
        max_reconnects=max(0, args.max_reconnects),
    )
    stream_b = ResilientCapture(
        source_b,
        "B",
        cap_b,
        args.open_attempts,
        max_reconnects=max(0, args.max_reconnects),
    )

    print("Dang chay tracking Global ID. Nhan 'q' hoac ESC de thoat.")
    start = time.time()
    frame_idx = 0
    last_tracks_a = []
    last_tracks_b = []
    last_states_a: dict[int, str] = {}
    last_states_b: dict[int, str] = {}
    window_a_shown = False
    window_b_shown = False

    try:
        while True:
            now_s = time.time() - start
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

            batch_detections: dict[str, list] = {}
            if process_frame:
                batch_names: list[str] = []
                batch_images: list[np.ndarray] = []
                if ret_a and frame_a is not None:
                    batch_names.append("A")
                    batch_images.append(frame_a)
                if ret_b and frame_b is not None:
                    batch_names.append("B")
                    batch_images.append(frame_b)
                batch_detections = dict(
                    zip(batch_names, detector.detect_batch(batch_images), strict=True)
                )

            # --- Channel A: ByteTrack (ngan han) + Global ID (toan cuc) ---
            if ret_a and frame_a is not None:
                if process_frame:
                    tracks_a = tracker_a.update(batch_detections["A"])
                    tracks_a = [t for t in tracks_a if t.bbox.area >= args.min_area]
                    confirmed_a = manager.update(
                        channel="A",
                        frame=frame_a,
                        tracks=[t for t in tracks_a if t.confirmed],
                        now_s=now_s,
                    )
                    last_tracks_a = confirmed_a
                    count_a.update(len(confirmed_a))
                    states_a = business_a.update(
                        now_s, {t.track_id for t in confirmed_a}
                    )
                    last_states_a = {g: s.value for g, s in states_a.items()}
                else:
                    confirmed_a = last_tracks_a
                if args.display:
                    vis = frame_a.copy()
                    draw_person_tracks(vis, confirmed_a, display_count=count_a.value,
                                       title="Global")
                    draw_business_states(vis, confirmed_a, last_states_a)
                    cv2.imshow("Channel A - tracking", vis)
                    window_a_shown = True

            # --- Channel B: ByteTrack (ngan han) + Global ID (toan cuc) ---
            if ret_b and frame_b is not None:
                if process_frame:
                    tracks_b = tracker_b.update(batch_detections["B"])
                    tracks_b = [t for t in tracks_b if t.bbox.area >= args.min_area]
                    confirmed_b = manager.update(
                        channel="B",
                        frame=frame_b,
                        tracks=[t for t in tracks_b if t.confirmed],
                        now_s=now_s,
                    )
                    last_tracks_b = confirmed_b
                    count_b.update(len(confirmed_b))
                    states_b = business_b.update(
                        now_s, {t.track_id for t in confirmed_b}
                    )
                    last_states_b = {g: s.value for g, s in states_b.items()}
                else:
                    confirmed_b = last_tracks_b
                if args.display:
                    vis_b = frame_b.copy()
                    draw_person_tracks(vis_b, confirmed_b, display_count=count_b.value,
                                       title="Global")
                    draw_business_states(vis_b, confirmed_b, last_states_b)
                    cv2.imshow("Channel B - tracking", vis_b)
                    window_b_shown = True

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
            frame_idx += 1
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
    finally:
        stream_a.close()
        stream_b.close()
        cv2.destroyAllWindows()
    elapsed = time.time() - start
    gids = sorted(manager.identities)
    print(f"Xong sau {elapsed:.1f}s. Count A: {count_a.value} | Count B: {count_b.value}")
    print(f"Global IDs: {gids}")
    for gid in gids:
        record = manager.identities[gid]
        print(f"  G{gid}: state={record.state.value} last_channel={record.channel} "
              f"hits={record.total_hits} gallery={len(record.gallery)}")


if __name__ == "__main__":
    main()
