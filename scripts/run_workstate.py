"""Demo MVP: detect + track 2 channel IMOU, Re-ID histogram, state machine.

Channel A (làm việc): YOLO detect -> IoU track -> kiểm tra ROI ghế.
  Rời ROI > leave_grace_s -> AWAY_SHORT + lưu embedding lúc rời.
Channel B (hành lang): YOLO detect -> IoU track -> trích embedding + zone.
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
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
from dotenv import load_dotenv

from camera_tracking.config import load_config
from camera_tracking.detection import YoloPersonDetector
from camera_tracking.domain import BoundingBox
from camera_tracking.tracking import IoUTracker
from camera_tracking.workstate import (
    CorridorZones,
    HistogramEmbedding,
    SeatZone,
    WorkStateConfig,
    WorkStateEngine,
)

load_dotenv()
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def imou_url(channel: int, subtype: int = 1) -> str | None:
    ip = os.getenv("IMOU_IP", "")
    user = os.getenv("IMOU_USER", "")
    password = os.getenv("IMOU_PASSWORD", "")
    if not ip or not user or not password:
        return None
    pw = quote(password, safe="")
    return f"rtsp://{user}:{pw}@{ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP workstate 2 camera (1 nguoi/ghe).")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--source-a", default=None, help="Override camera A.")
    parser.add_argument("--source-b", default=None, help="Override camera B.")
    parser.add_argument("--channel-a", type=int, default=1, help="Kenh IMOU cho cam A.")
    parser.add_argument("--channel-b", type=int, default=2, help="Kenh IMOU cho cam B.")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--event-log", type=Path, default=Path("output/workstate_events.jsonl"))
    parser.add_argument("--min-area", type=float, default=2000.0,
                        help="Bo box nguoi nho hon nguong (px^2) de giam nhieu.")
    parser.add_argument("--max-persons-b", type=int, default=3,
                        help="Channel B chi xet toi da N nguoi lon nhat trong vung.")
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


def draw_zones(frame: np.ndarray, seats: list[SeatZone], corridor=None) -> np.ndarray:
    for seat in seats:
        pts = np.asarray(seat.polygon, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cv2.putText(frame, seat.seat_id, tuple(map(int, seat.polygon[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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

    source_a = args.source_a or imou_url(args.channel_a) or config.camera.source
    source_b = args.source_b or imou_url(args.channel_b)
    if source_b is None:
        print("Thiếu --source-b và không có IMOU_* trong .env. "
              "Ví dụ: --source-b data/samples/hallway.mp4")
        raise SystemExit(2)
    print(f"Camera A: {source_a}\nCamera B: {source_b}")

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
        confidence=config.detection.confidence_threshold,
        person_class_id=config.detection.person_class_id,
        image_size=config.detection.image_size,
        device=config.detection.device,
    )
    tracker_a = IoUTracker(config.tracking.iou_threshold,
                           config.tracking.max_lost_frames, config.tracking.min_hits)
    tracker_b = IoUTracker(config.tracking.iou_threshold,
                           config.tracking.max_lost_frames, config.tracking.min_hits)
    embedder = HistogramEmbedding()

    cap_a = cv2.VideoCapture(source_a, cv2.CAP_FFMPEG)
    cap_b = cv2.VideoCapture(source_b, cv2.CAP_FFMPEG)
    if not cap_a.isOpened():
        print(f"Không mở được camera A: {source_a}")
    if not cap_b.isOpened():
        print(f"Không mở được camera B: {source_b}")
    if not cap_a.isOpened() and not cap_b.isOpened():
        raise SystemExit(1)

    args.event_log.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(args.event_log, "w", encoding="utf-8")
    print("Đang chạy MVP. Nhấn 'q' hoặc ESC để thoát.")
    start = time.time()
    frame_idx = 0

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
            ret_a, frame_a = cap_a.read() if cap_a.isOpened() else (False, None)
            ret_b, frame_b = cap_b.read() if cap_b.isOpened() else (False, None)
            if not ret_a and not ret_b:
                print("Hết stream cả 2 camera.")
                break

            # --- Channel A: mỗi ghế chỉ giữ 1 người (box lớn nhất trong ROI) ---
            if ret_a and frame_a is not None:
                tracks_a = tracker_a.update(detector.detect(frame_a))
                # Lọc box rác:-confidence đã lọc ở YOLO, thêm lọc diện tích tối thiểu.
                tracks_a = [t for t in tracks_a if t.bbox.area >= args.min_area]
                confirmed_a = [t for t in tracks_a if t.confirmed] or tracks_a
                presence: dict[str, bool] = {}
                embs_a: dict[str, np.ndarray | None] = {}
                best_per_seat = {}
                for seat in seats:
                    inside = [t for t in confirmed_a if seat.contains(t.bbox)]
                    if not inside:
                        presence[seat.seat_id] = False
                        continue
                    # Giới hạn 1 người/ghế: lấy box lớn nhất.
                    best = max(inside, key=lambda t: t.bbox.area)
                    best_per_seat[seat.seat_id] = best
                    presence[seat.seat_id] = True
                    embs_a[seat.seat_id] = embedder.extract(crop_of(frame_a, best.bbox))
                if seats:
                    log_events(engine.update_office(now, presence, embs_a))
                if args.display:
                    vis = draw_zones(frame_a.copy(), seats)
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

            # --- Channel B: chỉ xét người trong vùng + tối đa N người lớn nhất ---
            if ret_b and frame_b is not None:
                tracks_b = tracker_b.update(detector.detect(frame_b))
                tracks_b = [t for t in tracks_b if t.bbox.area >= args.min_area]
                confirmed_b = [t for t in tracks_b if t.confirmed] or tracks_b
                zoned = [(t, corridor.locate(t.bbox)) for t in confirmed_b]
                zoned = [(t, z) for t, z in zoned if z in ("hallway", "exit")]
                # Giới hạn số người để giảm nhiễu + nhẹ Re-ID.
                zoned.sort(key=lambda tz: tz[0].bbox.area, reverse=True)
                zoned = zoned[:args.max_persons_b]
                cand_embs, cand_zones = [], []
                for t, z in zoned:
                    cand_embs.append(embedder.extract(crop_of(frame_b, t.bbox)))
                    cand_zones.append(z)
                if cand_embs:
                    log_events(engine.update_corridor(now, cand_embs, cand_zones))
                if args.display:
                    vis_b = draw_zones(frame_b.copy(), [], corridor)
                    cv2.imshow("Channel B - hallway", vis_b)

            log_events(engine.tick(now))

            if args.display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("Đã nhấn phím thoát.")
                    break
            frame_idx += 1
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
    finally:
        cap_a.release()
        cap_b.release()
        cv2.destroyAllWindows()
        log_file.close()
    print(f"Xong. Trạng thái cuối: { {s: engine.state_of(s).value for s in engine.seat_ids} }")
    print(f"Event log: {args.event_log}")


if __name__ == "__main__":
    main()
