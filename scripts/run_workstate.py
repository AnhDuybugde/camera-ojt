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
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")
# Low-latency RTSP: TCP transport + no buffering + low delay decode.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
)

import cv2
import numpy as np

from camera_tracking.config import load_config
from camera_tracking.detection import YoloPersonDetector, resolve_device
from camera_tracking.tracking import (
    ByteTrackTracker,
    GlobalIdentityConfig,
    GlobalIdentityManager,
    IdentityState,
    StablePersonCount,
)
from camera_tracking.visualization import draw_global_labels, draw_person_tracks
from camera_tracking.workstate import (
    LABEL_UNKNOWN,
    ChannelBusinessTracker,
    HistogramEmbedding,
    IdentityReconciler,
    RoomPresenceAggregator,
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
        "--subtype-a", type=int, default=1, choices=(0, 1), help="0=main, 1=sub stream."
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
    parser.add_argument("--away-grace-s", type=float, default=3.0,
                        help="Business: absent/seat-leave over N sec -> AWAY.")
    parser.add_argument("--out-after-s", type=float, default=20.0,
                        help="Business: absent over N sec -> POSSIBLY_OUT.")
    parser.add_argument("--return-stable-s", type=float, default=2.0,
                        help="Business: stable presence N sec -> WORKING.")
    parser.add_argument("--new-track-conf", type=float, default=0.4,
                        help="Min YOLO score that may spawn a NEW tracklet "
                        "(ghosts below this never become Global IDs).")
    parser.add_argument("--move-ratio", type=float, default=None,
                        help="Interim displacement threshold 0..1 (default from "
                        "workstate.move_ratio, 0 = presence-only).")
    parser.add_argument("--stream-port", type=int, default=8765,
                        help="Live MJPEG port for the dashboard (0 disables).")
    parser.add_argument("--stream-host", default="127.0.0.1",
                        help="Live MJPEG bind host.")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable the live dashboard stream.")
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu / auto (mac dinh lay theo config).")
    parser.add_argument("--no-face", action="store_true",
                        help="Tat nhan dien khuon mat channel B (chi tracking).")
    parser.add_argument("--face-threshold", type=float, default=None,
                        help="Override face.match_threshold trong config.")
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
        except Exception:  # noqa: BLE001 - bad calibration, skip this point
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
    if bbox.x2 < 0 or bbox.y2 < 0 or bbox.x1 > frame_width or bbox.y1 > frame_height:
        return False
    return True


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


def _flush_queue(queue, supabase) -> tuple[int, int]:
    """Day pending writes len Supabase. Tra (ok, fail). Gap loi thi dung lai."""
    ok = fail = 0
    for row_id, kind, payload, _attempts in queue.peek(100):
        done = False
        if kind == "attendance":
            done = supabase.upsert_attendance(payload)
        elif kind == "room_status":
            done = supabase.upsert_room_status(payload)
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
            break
    return ok, fail


def _store_or_queue(queue, supabase, kind: str, payload: dict) -> None:
    """Ghi thang Supabase neu online, khong thi day vao queue (chong mat data)."""
    direct = False
    if supabase.ready or supabase.settings.enabled:
        if kind == "attendance":
            direct = supabase.upsert_attendance(payload)
        elif kind == "room_status":
            direct = supabase.upsert_room_status(payload)
        elif kind == "event":
            direct = supabase.insert_event(payload)
        elif kind == "face_crop":
            direct = supabase.upload_face_crop(
                payload.get("local_path", ""), payload.get("storage_path", "")
            )
    if not direct:
        queue.push(kind, payload)


def _merge_same_person_stale(
    day_cache, write_queue, supabase, gid_alias,
    gid_to_person, gid_to_display,
    day_str: str, canon_gid: int, person_id: str, display_name: str,
    manager, business_trackers=(),
) -> None:
    """Alias OTHER gids of the same face-matched person into canon_gid.

    Fixes the G1-OUT + G2-Working split-brain: when the same human was
    fragmented across IDs, the stale (non-ACTIVE) duplicate is backfilled
    with person info + merged_into so the dashboard hides it. A duplicate
    that is still ACTIVE (two bodies visible) is left alone: ambiguous.
    """
    for old_gid, pid in list(gid_to_person.items()):
        if pid != person_id or old_gid == canon_gid or old_gid in gid_alias:
            continue
        record = manager.identities.get(old_gid)
        if record is not None and record.state is IdentityState.ACTIVE:
            continue
        gid_alias[old_gid] = canon_gid
        gid_to_display[old_gid] = display_name
        for tracker in business_trackers:
            try:
                tracker.transfer_assignment(old_gid, canon_gid)
            except Exception:  # noqa: BLE001 - never break the loop
                pass
        cached = day_cache._status.get((day_str, old_gid))
        _store_or_queue(write_queue, supabase, "room_status", {
            "date": day_str, "global_id": old_gid,
            "person_id": person_id, "person_name": display_name,
            "in_room": cached.in_room if cached else True,
            "label": cached.label if cached else LABEL_UNKNOWN,
            "merged_into": canon_gid,
        })
        print(f"[Reconcile] stale G{old_gid} -> G{canon_gid} {display_name}")


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
            except Exception:  # noqa: BLE001 - never break the loop
                pass
        cached = day_cache._status.get((day_str, old_gid))
        _store_or_queue(write_queue, supabase, "room_status", {
            "date": day_str, "global_id": old_gid,
            "person_id": person_id, "person_name": display_name,
            "in_room": cached.in_room if cached else True,
            "label": cached.label if cached else LABEL_UNKNOWN,
            "merged_into": canon_gid,
        })
        try:
            supabase.reassign_face_crops(day_str, owner, person_id)
        except Exception:  # noqa: BLE001 - best effort, files stay put
            pass
        reconciler.forget(owner)
        print(f"[Reconcile] G{old_gid} ({owner}) -> G{canon_gid} {display_name}")


def _prune_identity_maps(
    manager, gid_to_person, gid_to_display, gid_to_score,
    unknown_of_gid, reconciler, fusion, business_trackers=(),
) -> None:
    """Drop per-ID memory for identities the manager already retired.

    Dead ghost IDs must not linger in overlay memory, seat assignment or
    the dashboard live list. DB history rows are kept (audit).
    """
    alive = set(manager.identities)
    for store in (gid_to_person, gid_to_display, gid_to_score):
        for gid in [g for g in store if g not in alive]:
            store.pop(gid, None)
    for gid in [g for g in unknown_of_gid if g not in alive]:
        reconciler.forget(unknown_of_gid.pop(gid))
    for tracker in business_trackers:
        tracker.forget_retired(alive)
    for gid in [g for g in list(fusion._was_out) if g not in alive]:
        fusion.forget(gid)


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
        except Exception:  # noqa: BLE001 - inspection only, never fatal
            pass


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
        # Precision-first: feed exactly the configured confidence (0.4).
        # No low-conf recovery: brief occlusions may fragment IDs instead
        # of risking misassignment. Reconcile + prune clean up afterwards.
        confidence=config.detection.confidence_threshold,
        person_class_id=config.detection.person_class_id,
        image_size=image_size,
        device=device,
    )
    effective_fps = max(1, round(config.camera.fps / config.camera.process_every_n_frames))
    tracker_a = ByteTrackTracker(
        frame_rate=effective_fps,
        track_buffer=30,
        track_high_threshold=args.new_track_conf,
        track_low_threshold=args.new_track_conf,
        new_track_threshold=args.new_track_conf,
        match_threshold=0.9,
        min_hits=config.tracking.min_hits,
    )
    tracker_b = ByteTrackTracker(
        frame_rate=effective_fps,
        track_buffer=30,
        track_high_threshold=args.new_track_conf,
        track_low_threshold=args.new_track_conf,
        new_track_threshold=args.new_track_conf,
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
        prune_after_s=ws_cfg.prune_after_s,
    )
    count_a = StablePersonCount(rise_frames=3, fall_frames=12)
    count_b = StablePersonCount(rise_frames=3, fall_frames=12)

    # --- Face diem danh (channel B) + daily store + room fusion ---
    # Tat ca optional: thieu model/lib/.env van chay tracking nhu cu.
    face_cfg = config.face
    face_threshold = args.face_threshold or face_cfg.match_threshold
    use_face = bool(face_cfg.enabled and not args.no_face)
    face_embedder = face_matcher = face_gallery = attendance = None
    gid_to_person: dict[int, str] = {}
    gid_to_display: dict[int, str] = {}
    gid_to_score: dict[int, float] = {}
    unknown_of_gid: dict[int, str] = {}
    unknown_counter = 0
    if use_face:
        try:
            from camera_tracking.face import (
                FaceAttendanceService,
                FaceMatcher,
                InsightFaceEmbedder,
                load_gallery,
            )
            try:
                face_embedder = InsightFaceEmbedder(
                    model_pack=face_cfg.model_pack, det_size=face_cfg.det_size
                )
                # Khong load model nang o day; load lazy o frame dau co nguoi.
            except Exception as error:  # noqa: BLE001
                print(f"Face disabled (khoi tao embedder loi: {error})")
                face_embedder = None
                use_face = False
            if use_face:
                face_gallery = load_gallery(
                    face_cfg.gallery_dir, face_embedder, face_cfg.name_map
                )
                face_matcher = FaceMatcher(face_gallery, threshold=face_threshold)
                attendance = FaceAttendanceService(
                    debounce_hits=config.attendance.debounce_hits,
                    window_s=config.attendance.window_s,
                    active_hour_start=config.attendance.active_hour_start,
                    active_hour_end=config.attendance.active_hour_end,
                )
                print(f"Face gallery: {len(face_gallery)} nguoi tu {face_cfg.gallery_dir} "
                      f"| threshold={face_threshold}")
        except ImportError as error:
            print(f"Face disabled (thieu module: {error})")
            use_face = False

    from camera_tracking.face import face_sharpness
    from camera_tracking.store.daily import DailyStateCache
    from camera_tracking.store.faces import FaceCropSaver
    from camera_tracking.store.queue import WriteQueue
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
    if config.store.supabase_enabled:
        if supabase.connect():
            print("Supabase: connected.")
        else:
            print(f"Supabase: local-only ({supabase.last_error or 'thieu SUPABASE_URL/KEY'})")
    fusion = RoomPresenceAggregator(
        leave_confirm_window_s=config.room_fusion.leave_confirm_window_s
    )
    room_status_now: dict = {}
    last_flush_s = 0.0
    # Unknown GID -> canonical (identified) GID after face reconcile.
    gid_alias: dict[int, int] = {}
    reconciler = IdentityReconciler(threshold=face_threshold)

    # --- Live MJPEG stream for the dashboard (2 cameras + model overlay) ---
    stream_on = False
    streamer = None
    if not args.no_stream and args.stream_port:
        from camera_tracking.streaming import MjpegStreamer
        streamer = MjpegStreamer(host=args.stream_host, port=args.stream_port)
        stream_on = streamer.start()
        if stream_on:
            print(f"Live stream: http://{args.stream_host}:{args.stream_port} "
                  f"(dashboard Live tab, VITE_STREAM_URL)")
        else:
            print(f"Stream port {args.stream_port} busy, "
                  f"running without live stream.")

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
    last_push_a = 0.0
    last_push_b = 0.0
    seen_gids: set[int] = set()
    ghosts_dir = Path(config.output.output_dir) / "ghosts"
    face_marks_b: list = []  # (x1,y1,x2,y2,score,known) for cam B overlay

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
                face_marks_b.clear()
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
                    tracks_a = [t for t in tracks_a
                                if t.bbox.area >= args.min_area
                                and _sane_box(t.bbox, config.camera.frame_width,
                                              config.camera.frame_height)]
                    confirmed_a = manager.update(
                        channel="A",
                        frame=frame_a,
                        tracks=[t for t in tracks_a if t.confirmed],
                        now_s=now_s,
                    )
                    last_tracks_a = confirmed_a
                    count_a.update(len(confirmed_a))
                    if use_roi:
                        floor_a = _track_floor_points(confirmed_a, projector)
                    else:
                        # Interim: normalized centers, displacement mode.
                        floor_a = _track_norm_centers(
                            confirmed_a, config.camera.frame_width,
                            config.camera.frame_height)
                    states_a = business_a.update(
                        now_s, {t.track_id for t in confirmed_a},
                        floor_a, gid_to_person,
                    )
                    last_states_a = {g: s.value for g, s in states_a.items()}
                else:
                    confirmed_a = last_tracks_a

            # --- Channel B: ByteTrack + Global ID + Face diem danh ---
            if ret_b and frame_b is not None:
                if process_frame:
                    tracks_b = tracker_b.update(batch_detections["B"])
                    tracks_b = [t for t in tracks_b
                                if t.bbox.area >= args.min_area
                                and _sane_box(t.bbox, config.camera.frame_width,
                                              config.camera.frame_height)]
                    confirmed_b = manager.update(
                        channel="B",
                        frame=frame_b,
                        tracks=[t for t in tracks_b if t.confirmed],
                        now_s=now_s,
                    )
                    last_tracks_b = confirmed_b
                    count_b.update(len(confirmed_b))
                    states_b = business_b.update(
                        now_s, {t.track_id for t in confirmed_b},
                        None, gid_to_person,
                    )
                    last_states_b = {g: s.value for g, s in states_b.items()}

                    if use_face and attendance is not None and face_matcher is not None:
                        now_dt = datetime.now()
                        day_str = now_dt.date().isoformat()
                        wall_iso = now_dt.isoformat(timespec="seconds")
                        time_tag = now_dt.strftime("%H%M%S")
                        face_tick = (frame_idx // max(1, config.camera.process_every_n_frames)) \
                            % max(1, face_cfg.process_every_k) == 0
                        if face_tick:
                            for track in confirmed_b:
                                if track.bbox.area < face_cfg.min_person_area_px:
                                    continue
                                crop = _person_crop(frame_b, track.bbox)
                                if crop is None:
                                    continue
                                # Crop origin in frame coords (same clamp as
                                # _person_crop) to draw face boxes on vis_b.
                                _bh, _bw = frame_b.shape[:2]
                                _ox = max(0, min(_bw, round(track.bbox.x1)))
                                _oy = max(0, min(_bh, round(track.bbox.y1)))
                                try:
                                    dets = face_embedder.detect_embed(crop)
                                except RuntimeError:
                                    dets = []
                                if not dets:
                                    continue
                                det = dets[0]
                                fw = det.bbox[2] - det.bbox[0]
                                fh = det.bbox[3] - det.bbox[1]
                                if min(fw, fh) < face_cfg.min_face_px:
                                    continue
                                fx1 = max(0, int(det.bbox[0])); fy1 = max(0, int(det.bbox[1]))
                                fx2 = min(crop.shape[1], int(det.bbox[2]))
                                fy2 = min(crop.shape[0], int(det.bbox[3]))
                                face_img = crop[fy1:fy2, fx1:fx2] \
                                    if fx2 > fx1 and fy2 > fy1 else None
                                sharp = face_sharpness(face_img)
                                if sharp < face_cfg.min_blur_variance:
                                    continue
                                match = face_matcher.match(det.embedding)
                                # Live feedback: face box + score on cam B
                                # (green = known, red = unknown/low score).
                                face_marks_b.append(
                                    (_ox + fx1, _oy + fy1, _ox + fx2, _oy + fy2,
                                     match.score, match.is_known))
                                if match.is_known and match.person is not None:
                                    gid_to_score[track.track_id] = match.score
                                    gid_to_person[track.track_id] = match.person.person_id
                                    # Show the name as soon as the face matches
                                    # (DB tick stays debounced below).
                                    gid_to_display[track.track_id] = \
                                        match.person.display_name
                                    ticked = attendance.observe(
                                        day=day_str, global_id=track.track_id,
                                        person_id=match.person.person_id,
                                        display_name=match.person.display_name,
                                        score=match.score, now_s=now_s,
                                        wall_time_iso=wall_iso,
                                    )
                                    if ticked is not None:
                                        if day_cache.attendance_should_write(
                                                day_str, ticked.person_id):
                                            _store_or_queue(write_queue, supabase, "attendance", {
                                                "date": day_str, "person_id": ticked.person_id,
                                                "person_name": ticked.display_name,
                                                "global_id": ticked.global_id,
                                                "attended": True, "check_in_at": wall_iso,
                                                "face_score": ticked.face_score,
                                                "needs_review": False,
                                            })
                                            _store_or_queue(write_queue, supabase, "event", {
                                                "date": day_str, "global_id": ticked.global_id,
                                                "person_id": ticked.person_id,
                                                "event": "CHECK_IN", "channel": "B",
                                                "at": wall_iso,
                                            })
                                            print(f"[Diem danh] {ticked.display_name} "
                                                  f"(G{ticked.global_id}, "
                                                  f"score={ticked.face_score:.2f})")
                                    # Reconcile: stale unknown GIDs that match
                                    # this person merge into the current gid.
                                    if match.person.embedding is not None:
                                        _reconcile_unknowns(
                                            day_cache, write_queue, supabase,
                                            reconciler, gid_alias, gid_to_person,
                                            gid_to_display, unknown_of_gid,
                                            day_str, track.track_id,
                                            match.person.person_id,
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
                                        track.track_id,
                                        match.person.person_id,
                                        match.person.display_name,
                                        manager, (business_a, business_b),
                                    )
                                    local = crop_saver.save_best_crop(
                                        day=day_str, owner=match.person.person_id,
                                        known=True, global_id=track.track_id,
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
                else:
                    confirmed_b = last_tracks_b

            # --- Fuse trang thai phong A+B (moi frame xu ly) ---
            if process_frame:
                now_dt = datetime.now()
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
                    established = (record is not None and record.total_hits
                                   >= ws_cfg.tentative_min_hits)
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
                    if write_row and established:
                        row = day_cache._status[(day_str, wgid)]
                        _store_or_queue(write_queue, supabase, "room_status", {
                            "date": day_str, "global_id": wgid,
                            "person_id": gid_to_person.get(wgid),
                            "person_name": gid_to_display.get(wgid),
                            "in_room": st.in_room, "label": st.label,
                            "last_leave_at": row.last_leave_at,
                            "last_enter_at": row.last_enter_at,
                        })
                if now_s - last_flush_s >= config.room_fusion.flush_s:
                    last_flush_s = now_s
                    _flush_queue(write_queue, supabase)
                    _prune_identity_maps(
                        manager, gid_to_person, gid_to_display,
                        gid_to_score, unknown_of_gid, reconciler,
                        fusion, (business_a, business_b),
                    )

            # --- Live view: annotated Global-ID frames for window/stream ---
            need_vis = args.display or stream_on
            vis_a = vis_b = None
            if ret_a and frame_a is not None and need_vis:
                vis_a = frame_a.copy()
                draw_person_tracks(vis_a, last_tracks_a, display_count=count_a.value,
                                   title="Global", status=room_status_now)
                draw_global_labels(vis_a, last_tracks_a, room_status_now, gid_to_display)
            if ret_b and frame_b is not None and need_vis:
                vis_b = frame_b.copy()
                draw_person_tracks(vis_b, last_tracks_b, display_count=count_b.value,
                                   title="Global", status=room_status_now)
                draw_global_labels(vis_b, last_tracks_b, room_status_now, gid_to_display)
                for x1m, y1m, x2m, y2m, fscore, fknown in face_marks_b:
                    fcolor = (40, 180, 40) if fknown else (60, 60, 220)
                    cv2.rectangle(vis_b, (int(x1m), int(y1m)),
                                  (int(x2m), int(y2m)), fcolor, 2)
                    cv2.putText(vis_b, f"{fscore:.2f}",
                                (int(x1m), max(0, int(y1m) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fcolor, 2)
            if args.display:
                if vis_a is not None:
                    cv2.imshow("Channel A - tracking", vis_a)
                    window_a_shown = True
                if vis_b is not None:
                    cv2.imshow("Channel B - tracking", vis_b)
                    window_b_shown = True
            if stream_on and process_frame:
                now_push = time.time()
                if vis_a is not None and now_push - last_push_a >= 0.1:
                    ok_a, buf_a = cv2.imencode(
                        ".jpg", vis_a, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok_a:
                        streamer.push("cam_a", buf_a.tobytes())
                        last_push_a = now_push
                if vis_b is not None and now_push - last_push_b >= 0.1:
                    ok_b, buf_b = cv2.imencode(
                        ".jpg", vis_b, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok_b:
                        streamer.push("cam_b", buf_b.tobytes())
                        last_push_b = now_push
                streamer.set_status({
                    "people": [
                        {"gid": gid,
                         "name": st.display_name,
                         "label": st.label,
                         "in_room": st.in_room}
                        for gid, st in sorted(room_status_now.items())
                    ],
                    "count_a": count_a.value,
                    "count_b": count_b.value,
                })

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
        try:
            _flush_queue(write_queue, supabase)
        except Exception as error:  # noqa: BLE001
            print(f"Flush cuoi loi (da giu trong queue): {error}")
        if streamer is not None:
            streamer.stop()
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
    ghost_hits = max(10, 2 * ws_cfg.tentative_min_hits)
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
        try:
            day_now = datetime.now().date().isoformat()
            ticked = attendance.records_for_day(day_now)
            print(f"Diem danh {day_now}: {len(ticked)} nguoi")
            for rec in ticked:
                print(f"  {rec.display_name} (G{rec.global_id}, "
                      f"score={rec.face_score:.2f}, in={rec.wall_time})")
            if unknown_of_gid:
                print(f"Mat la: {len(unknown_of_gid)} ({', '.join(sorted(unknown_of_gid.values()))})")
        except Exception:
            pass
    try:
        pending = len(write_queue)
        if pending:
            print(f"Queue ton {pending} writes (offline/miss env) -> se flush lan chay sau.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
