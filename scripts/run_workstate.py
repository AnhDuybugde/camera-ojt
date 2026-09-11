"""Demo MVP: detect + track 2 channel IMOU, Re-ID histogram, state machine.

Channel A (làm việc): YOLO -> ByteTrack -> persistent ID -> kiểm tra ROI ghế.
  Rời ROI > leave_grace_s -> AWAY_SHORT + lưu embedding lúc rời.
Channel B (hành lang): YOLO -> ByteTrack -> persistent ID -> embedding + zone.
  Khớp với AWAY_SHORT trong 5 phút -> RESTROOM, quá timeout -> OUT_OF_OFFICE.

Chạy:
  python scripts/run_workstate.py --config config/default.yaml --display
  python scripts/run_workstate.py --source-a 0 --source-b data/samples/hallway.mp4 --display

RTSP IMOU lấy từ .env (IMOU_IP/USER/PASSWORD) nếu không truyền --source.
Nhấn 'q' hoặc ESC để thoát.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import ExitStack
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
from camera_tracking.domain import BoundingBox
from camera_tracking.tracking import (
    ByteTrackTracker,
    PersistentIdentityTracker,
    StablePersonCount,
)
from camera_tracking.visualization import draw_person_tracks
from camera_tracking.workstate import (
    CorridorZones,
    HistogramEmbedding,
    SeatZone,
    WorkStateConfig,
    WorkStateEngine,
)

reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
if reconfigure_stdout is not None:
    try:
        reconfigure_stdout(encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"Khong the dat UTF-8 cho terminal: {error}", file=sys.stderr)


def normalize_source(src) -> int | str:
    """'0' -> 0 (webcam), '1' -> 1, còn lại giữ nguyên (file/RTSP)."""
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
        # Webcam trên Windows cần DSHOW, không dùng FFMPEG.
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
    parser = argparse.ArgumentParser(description="MVP workstate 2 camera (1 nguoi/ghe).")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--source-a", default=None, help="Override camera A.")
    parser.add_argument("--source-b", default=None, help="Override camera B.")
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
    parser.add_argument("--event-log", type=Path, default=Path("output/workstate_events.jsonl"))
    parser.add_argument("--min-area", type=float, default=2000.0,
                        help="Bo box nguoi nho hon nguong (px^2) de giam nhieu.")
    parser.add_argument("--max-persons-b", type=int, default=3,
                        help="Channel B chi xet toi da N nguoi lon nhat trong vung.")
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu / auto (mac dinh lay theo config).")
    return parser.parse_args()


def crop_of(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox.x1))
    y1 = max(0, int(bbox.y1))
    x2 = min(w, int(bbox.x2))
    y2 = min(h, int(bbox.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def normalize_frame_size(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Put every source in the coordinate system used by configured ROI polygons."""
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    interpolation = cv2.INTER_AREA if frame.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def draw_zones(
    frame: np.ndarray,
    seats: list[SeatZone],
    corridor=None,
    ambiguous_seats: set[str] | None = None,
) -> np.ndarray:
    ambiguous_seats = ambiguous_seats or set()
    for seat in seats:
        ambiguous = seat.seat_id in ambiguous_seats
        color = (0, 0, 255) if ambiguous else (0, 255, 0)
        label = f"{seat.seat_id}:AMBIGUOUS" if ambiguous else seat.seat_id
        pts = np.asarray(seat.polygon, dtype=np.int32)
        cv2.polylines(frame, [pts], True, color, 2)
        cv2.putText(
            frame,
            label,
            tuple(map(int, seat.polygon[0])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    if corridor is not None:
        for poly, color, label in [
            (corridor.hallway, (255, 0, 0), "hallway"),
            (corridor.exit_door, (0, 0, 255), "exit"),
        ]:
            if len(poly) >= 3:
                pts = np.asarray(poly, dtype=np.int32)
                cv2.polylines(frame, [pts], True, color, 2)
                cv2.putText(frame, label, tuple(map(int, poly[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ws = config.workstate

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
    print(
        f"Device: {device} | Camera A: {source_label(source_a)} | "
        f"Camera B: {source_label(source_b)}"
    )

    seats = [SeatZone(s.seat_id, s.name or s.seat_id, [tuple(p) for p in s.polygon], s.channel)
             for s in ws.seats]
    corridor = CorridorZones(hallway=[tuple(p) for p in ws.corridor.hallway],
                             exit_door=[tuple(p) for p in ws.corridor.exit_door])
    engine = WorkStateEngine(
        seat_ids=[s.seat_id for s in seats] or ["seat_01"],
        config=WorkStateConfig(
            leave_grace_s=ws.leave_grace_s,
            corridor_match_window_s=ws.corridor_match_window_s,
            restroom_return_window_s=ws.restroom_return_window_s,
            similarity_threshold=ws.similarity_threshold,
        ),
    )
    if not seats:
        print("Cảnh báo: chưa cấu hình ghế nào, dùng seat_01 mặc định (không có ROI).")

    detector = YoloPersonDetector(
        model_path=config.detection.model_path,
        # ByteTrack uses weak detections to recover an existing person through occlusion.
        confidence=min(0.10, config.detection.confidence_threshold),
        person_class_id=config.detection.person_class_id,
        image_size=config.detection.image_size,
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
    identity_a = PersistentIdentityTracker(
        embedder,
        max_missing_frames=240,
        max_center_distance_ratio=0.35,
        match_threshold=0.34,
    )
    identity_b = PersistentIdentityTracker(
        embedder,
        max_missing_frames=120,
        max_center_distance_ratio=0.35,
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

    args.event_log.parent.mkdir(parents=True, exist_ok=True)
    resources = ExitStack()
    log_file = resources.enter_context(args.event_log.open("a", encoding="utf-8"))
    print("Đang chạy MVP. Nhấn 'q' hoặc ESC để thoát.")
    start = time.time()
    frame_idx = 0
    last_tracks_a = []
    last_tracks_b = []
    window_a_shown = False
    window_b_shown = False
    reported_ambiguous_seats: set[str] = set()

    def log_events(events) -> None:
        for event in events:
            line = (f"[{event.timestamp_s:7.1f}s] {event.seat_id}: "
                    f"{event.prev.value} -> {event.new.value} ({event.reason}"
                    + (f", score={event.score:.2f}" if event.score is not None else "")
                    + ")")
            print(line)
            log_file.write(json.dumps({
                "t": event.timestamp_s, "seat": event.seat_id,
                "prev": event.prev.value, "new": event.new.value,
                "reason": event.reason, "score": event.score,
            }, ensure_ascii=False) + "\n")
            log_file.flush()

    try:
        while True:
            now = time.time() - start
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

            # --- Channel A: mỗi ghế chỉ giữ 1 người (box lớn nhất trong ROI) ---
            if ret_a and frame_a is not None:
                if process_frame:
                    tracks_a = tracker_a.update(batch_detections["A"])
                    tracks_a = [t for t in tracks_a if t.bbox.area >= args.min_area]
                    confirmed_raw_a = [track for track in tracks_a if track.confirmed]
                    confirmed_a = identity_a.update(frame_a, confirmed_raw_a)
                    last_tracks_a = confirmed_a
                    count_a.update(len(confirmed_a))
                else:
                    confirmed_a = last_tracks_a
                presence: dict[str, bool] = {}
                embs_a: dict[str, np.ndarray | None] = {}
                best_per_seat = {}
                ambiguous_seats: set[str] = set()
                for seat in seats:
                    inside = [t for t in confirmed_a if seat.contains(t.bbox)]
                    if not inside:
                        presence[seat.seat_id] = False
                        continue
                    if len(inside) > 1:
                        ambiguous_seats.add(seat.seat_id)
                        if process_frame and seat.seat_id not in reported_ambiguous_seats:
                            print(
                                f"ROI {seat.seat_id} chua {len(inside)} nguoi; "
                                "tam dung cap nhat trang thai ROI nay."
                            )
                            reported_ambiguous_seats.add(seat.seat_id)
                        continue
                    # Giới hạn 1 người/ghế: lấy box lớn nhất.
                    best = max(inside, key=lambda t: t.bbox.area)
                    best_per_seat[seat.seat_id] = best
                    presence[seat.seat_id] = True
                    if process_frame:
                        embs_a[seat.seat_id] = embedder.extract(crop_of(frame_a, best.bbox))
                if seats and process_frame:
                    log_events(engine.update_office(now, presence, embs_a))
                if args.display:
                    vis = draw_zones(
                        frame_a.copy(),
                        seats,
                        ambiguous_seats=ambiguous_seats,
                    )
                    draw_person_tracks(vis, confirmed_a, display_count=count_a.value)
                    for seat in seats:
                        best = best_per_seat.get(seat.seat_id)
                        if best is not None:
                            x1, y1, x2, y2 = map(int, (best.bbox.x1, best.bbox.y1,
                                                       best.bbox.x2, best.bbox.y2))
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for sid in engine.seat_ids:
                        st = engine.statuses[sid].state.value
                        cv2.putText(vis, f"{sid}:{st}", (20, 30 + 25 * engine.seat_ids.index(sid)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Channel A - working", vis)
                    window_a_shown = True

            # --- Channel B: chỉ xét người trong vùng + tối đa N người lớn nhất ---
            if ret_b and frame_b is not None:
                if process_frame:
                    tracks_b = tracker_b.update(batch_detections["B"])
                    tracks_b = [t for t in tracks_b if t.bbox.area >= args.min_area]
                    confirmed_raw_b = [track for track in tracks_b if track.confirmed]
                    confirmed_b = identity_b.update(frame_b, confirmed_raw_b)
                    last_tracks_b = confirmed_b
                    count_b.update(len(confirmed_b))
                else:
                    confirmed_b = last_tracks_b
                zoned = [(t, corridor.locate(t.bbox)) for t in confirmed_b]
                zoned = [(t, z) for t, z in zoned if z in ("hallway", "exit")]
                # Giới hạn số người để giảm nhiễu + nhẹ Re-ID.
                zoned.sort(key=lambda tz: tz[0].bbox.area, reverse=True)
                zoned = zoned[:args.max_persons_b]
                cand_embs, cand_zones = [], []
                for t, z in zoned:
                    cand_embs.append(embedder.extract(crop_of(frame_b, t.bbox)))
                    cand_zones.append(z)
                if cand_embs and process_frame:
                    log_events(engine.update_corridor(now, cand_embs, cand_zones))
                if args.display:
                    vis_b = draw_zones(frame_b.copy(), [], corridor)
                    draw_person_tracks(vis_b, confirmed_b, display_count=count_b.value)
                    cv2.imshow("Channel B - hallway", vis_b)
                    window_b_shown = True

            log_events(engine.tick(now))

            if args.display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("Đã nhấn phím thoát.")
                    break
                window_a_closed = window_a_shown and cv2.getWindowProperty(
                    "Channel A - working", cv2.WND_PROP_VISIBLE
                ) < 1
                window_b_closed = window_b_shown and cv2.getWindowProperty(
                    "Channel B - hallway", cv2.WND_PROP_VISIBLE
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
        resources.close()
    print(f"Xong. Trạng thái cuối: { {s: engine.state_of(s).value for s in engine.seat_ids} }")
    print(f"Event log: {args.event_log}")


if __name__ == "__main__":
    main()
