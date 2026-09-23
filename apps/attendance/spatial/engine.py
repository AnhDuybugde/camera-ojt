"""Background spatial analysis worker, isolated from attendance recording."""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from camera.camera_manager import CameraManager
from face.detector import FaceDetector
from face.recognizer import FaceRecognizer
from spatial.detector import PersonDetector
from spatial.repository import SpatialRepository, local_now
from spatial.tracker import CentroidTracker, TrackSnapshot
from spatial.zones import Zone, zone_at

logger = logging.getLogger(__name__)


class SpatialAnalyticsEngine:
    def __init__(
        self, camera_id: str, camera: CameraManager, detector: PersonDetector,
        repository: SpatialRepository, zones: tuple[Zone, ...],
        face_detector: FaceDetector | None = None,
        recognizer: FaceRecognizer | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.camera = camera
        self.detector = detector
        self.repository = repository
        self.zones = zones
        self.face_detector = face_detector
        self.recognizer = recognizer
        self.tracker = CentroidTracker(
            max_distance=float(os.getenv("SPATIAL_TRACK_DISTANCE", "140")),
            max_missed=int(os.getenv("SPATIAL_TRACK_MAX_MISSED", "5")),
        )
        self.analysis_interval = max(0.2, float(os.getenv("SPATIAL_ANALYSIS_INTERVAL", "0.8")))
        self.sample_interval = max(1.0, float(os.getenv("SPATIAL_SAMPLE_INTERVAL", "5")))
        self.identity_interval = max(1.0, float(os.getenv("SPATIAL_IDENTITY_INTERVAL", "3")))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._tracks: list[TrackSnapshot] = []
        self._track_zones: dict[int, str | None] = {}
        self._visit_starts: dict[int, tuple[str, datetime]] = {}
        self.last_inference_ms = 0
        self.error = ""

    def start(self) -> "SpatialAnalyticsEngine":
        if self._thread and self._thread.is_alive():
            return self
        self.camera.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"spatial-{self.camera_id}",
        )
        self._thread.start()
        return self

    def _loop(self) -> None:
        last_sample = 0.0
        last_identity = 0.0
        while not self._stop.is_set():
            ok, frame = self.camera.read()
            if not ok or frame is None:
                self.error = self.camera.error or "Waiting for camera..."
                self._stop.wait(0.1)
                continue
            started = time.monotonic()
            try:
                detections = self.detector.detect(frame)
                tracks = self.tracker.update([item.box for item in detections])
                now_mono = time.monotonic()
                if now_mono - last_identity >= self.identity_interval:
                    self._identify(frame, tracks)
                    tracks = [self.tracker.snapshot(item.track_id) or item for item in tracks]
                    last_identity = now_mono
                self._update_zones(tracks, frame.shape[1], frame.shape[0])
                for retired in self.tracker.pop_retired():
                    self._close_visit(retired, local_now())
                    self._track_zones.pop(retired.track_id, None)
                counts = self._counts(tracks, frame.shape[1], frame.shape[0])
                if now_mono - last_sample >= self.sample_interval:
                    self.repository.record_density(self.camera_id, counts)
                    last_sample = now_mono
                annotated = self._draw(frame, tracks)
                with self._lock:
                    self._frame = annotated
                    self._tracks = tracks
                self.error = ""
            except Exception as exc:
                self.error = f"Spatial analytics error: {exc}"
                logger.exception("Spatial analytics worker failed for %s", self.camera_id)
            self.last_inference_ms = int((time.monotonic() - started) * 1000)
            self._stop.wait(self.analysis_interval)

    def _identify(self, frame: np.ndarray, tracks: list[TrackSnapshot]) -> None:
        if not tracks or self.face_detector is None or self.recognizer is None:
            return
        try:
            for face in self.face_detector.detect(frame):
                embedding = getattr(face, "embedding", None)
                if embedding is None:
                    continue
                match = self.recognizer.match(embedding)
                if not match.recognized:
                    continue
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                center = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)
                candidates = [
                    track for track in tracks
                    if track.box[0] <= center[0] <= track.box[2]
                    and track.box[1] <= center[1] <= track.box[3]
                ]
                if candidates:
                    track = min(candidates, key=lambda item: abs(item.anchor[0] - center[0]))
                    self.tracker.identify(track.track_id, match.employee_id, match.full_name)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("Identity binding skipped for %s: %s", self.camera_id, exc)

    def _update_zones(self, tracks: list[TrackSnapshot], width: int, height: int) -> None:
        now = local_now()
        for track in tracks:
            current = zone_at(self.zones, track.anchor[0] / width, track.anchor[1] / height)
            current_id = current.zone_id if current else None
            previous_id = self._track_zones.get(track.track_id)
            if track.track_id not in self._track_zones or current_id != previous_id:
                if previous_id is not None:
                    self._close_visit(track, now)
                self.repository.record_transition(
                    self.camera_id, track.track_id, track.employee_id, track.employee_name,
                    previous_id, current_id, now,
                )
                if current_id is not None:
                    self._visit_starts[track.track_id] = (current_id, now)
                self._track_zones[track.track_id] = current_id

    def _close_visit(self, track: TrackSnapshot, ended_at: datetime) -> None:
        visit = self._visit_starts.pop(track.track_id, None)
        if visit:
            zone_id, started_at = visit
            self.repository.record_visit(
                self.camera_id, track.track_id, track.employee_id, track.employee_name,
                zone_id, started_at, ended_at,
            )

    def _counts(self, tracks: list[TrackSnapshot], width: int, height: int) -> dict[str, int]:
        counts = Counter(
            zone.zone_id
            for track in tracks
            for zone in [zone_at(self.zones, track.anchor[0] / width, track.anchor[1] / height)]
            if zone is not None
        )
        return {zone.zone_id: counts[zone.zone_id] for zone in self.zones}

    def _draw(self, frame: np.ndarray, tracks: list[TrackSnapshot]) -> np.ndarray:
        output = frame.copy()
        height, width = output.shape[:2]
        overlay = output.copy()
        for zone in self.zones:
            polygon = np.array([(int(x * width), int(y * height)) for x, y in zone.points], np.int32)
            cv2.fillPoly(overlay, [polygon], zone.color)
            cv2.polylines(output, [polygon], True, zone.color, 2)
            x, y = polygon[0]
            cv2.putText(output, zone.label, (x + 8, max(24, y + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, zone.color, 2)
        output = cv2.addWeighted(overlay, 0.12, output, 0.88, 0)
        for track in tracks:
            x1, y1, x2, y2 = track.box
            known = track.employee_id is not None
            color = (16, 185, 129) if known else (79, 70, 229)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"#{track.track_id} · {track.employee_name if known else 'Chưa xác định'}"
            cv2.putText(output, label, (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if len(track.trail) > 1:
                trail = np.array(track.trail, np.int32).reshape((-1, 1, 2))
                cv2.polylines(output, [trail], False, color, 2)
        return output

    def latest(self) -> tuple[np.ndarray | None, list[TrackSnapshot]]:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return frame, list(self._tracks)

    def current_counts(self) -> dict[str, int]:
        with self._lock:
            tracks = list(self._tracks)
            shape = None if self._frame is None else self._frame.shape
        if shape is None:
            return {zone.zone_id: 0 for zone in self.zones}
        return self._counts(tracks, shape[1], shape[0])

    def track_zone(self, track_id: int) -> str | None:
        return self._track_zones.get(track_id)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=4)
        now = local_now()
        for track in self._tracks:
            self._close_visit(track, now)
        self.camera.stop()
