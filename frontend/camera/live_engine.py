"""Background recognition worker and annotated-frame producer."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from attendance.attendance_service import AttendanceService
from camera.camera_manager import CameraManager
from config import settings
from face.detector import FaceDetector
from face.embedding import blur_score
from face.liveness import LivenessDetector
from face.recognizer import FaceRecognizer
from face.temporal_identity import TemporalIdentityResolver
from spatial.activity_engine import ActivityRuntime, ActivityView, IdentityObservation
from spatial.tracker import CentroidTracker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectionView:
    box: tuple[int, int, int, int]
    employee_id: str | None
    name: str
    similarity: float
    attendance: str


class LiveAttendanceEngine:
    def __init__(
        self, camera: CameraManager, detector: FaceDetector,
        recognizer: FaceRecognizer, attendance: AttendanceService,
        activity: ActivityRuntime | None = None,
        recognition_camera: CameraManager | None = None,
    ) -> None:
        self.camera, self.detector = camera, detector
        self.recognition_camera = recognition_camera or camera
        self.recognizer, self.attendance = recognizer, attendance
        self.liveness = LivenessDetector()
        self.face_tracker = CentroidTracker(
            max_distance=settings.face_track_distance, max_missed=4,
        )
        self.identity_resolver = TemporalIdentityResolver(
            required_votes=settings.face_confirm_votes,
            confirm_ratio=settings.face_confirm_ratio,
            window_seconds=settings.face_vote_window_seconds,
            ttl_seconds=settings.face_identity_ttl_seconds,
        )
        self.activity = activity
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._detections: list[DetectionView] = []
        self.last_inference_ms = 0
        self.error = ""

    def start(self) -> "LiveAttendanceEngine":
        if self._thread and self._thread.is_alive():
            return self
        self.camera.start()
        if self.recognition_camera is not self.camera:
            self.recognition_camera.start()
        if self.activity is not None:
            try:
                self.activity.start()
            except Exception as exc:
                logger.warning("Activity start failed; attendance continues: %s", exc)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="face-recognition")
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.recognition_camera.read()
            if (not ok or frame is None) and self.recognition_camera is not self.camera:
                ok, frame = self.camera.read()
            if not ok or frame is None:
                self.error = self.camera.error or "Waiting for camera..."
                time.sleep(0.1)
                continue
            started = time.monotonic()
            try:
                detections = self._recognize(frame)
                detections = self._scale_for_display(detections, frame)
                if self.activity is not None:
                    try:
                        self.activity.submit_identities([
                            IdentityObservation(item.box, item.employee_id, item.name)
                            for item in detections if item.employee_id is not None
                        ])
                    except Exception as exc:
                        logger.warning("Activity identity binding skipped: %s", exc)
                self.error = ""
                with self._lock:
                    self._detections = detections
            except Exception as exc:
                self.error = f"Recognition error: {exc}"
                logger.exception("Recognition worker error")
            self.last_inference_ms = int((time.monotonic() - started) * 1000)
            # Bound CPU usage and give the second camera fair access to the shared model.
            self._stop.wait(settings.recognition_interval)

    def _recognize(self, frame: np.ndarray) -> list[DetectionView]:
        results: list[DetectionView] = []
        faces = [face for face in self.detector.detect(frame) if self._face_quality_ok(frame, face)]
        person_boxes = [item.box for item in self.activity.latest()] if self.activity is not None else []
        if settings.face_require_person_box and person_boxes:
            faces = [face for face in faces if self._face_belongs_to_person(face, person_boxes)]
        boxes = [tuple(map(int, face.bbox)) for face in faces]
        tracks = self.face_tracker.update(boxes)
        tracks_by_box = {track.box: track for track in tracks}
        now = time.monotonic()
        for face, box in zip(faces, boxes):
            embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue
            track = tracks_by_box.get(box)
            if track is None:
                continue
            candidate = self.recognizer.match(embedding)
            decision = self.identity_resolver.update(track.track_id, candidate, now)
            match = decision.match
            attendance_message = "UNKNOWN"
            if decision.status == "verifying":
                attendance_message = "VERIFYING"
                employee_id, full_name = None, "Đang xác minh"
            elif match.recognized:
                employee_id, full_name = match.employee_id, match.full_name
                live = self.liveness.check(frame, face)
                if live.is_live:
                    attendance_message = self.attendance.record(match.employee_id).message  # type: ignore[arg-type]
                    logger.info(
                        "Employee recognized: %s score=%.3f status=%s",
                        match.employee_id, match.similarity, decision.status,
                    )
                else:
                    attendance_message = "LIVENESS FAILED"
            else:
                employee_id, full_name = None, "UNKNOWN"
                logger.debug("Unknown face score=%.3f", candidate.similarity)
            results.append(DetectionView(
                box, employee_id, full_name, match.similarity, attendance_message,
            ))
        for retired in self.face_tracker.pop_retired():
            self.identity_resolver.remove(retired.track_id)
        return results

    @staticmethod
    def _face_quality_ok(frame: np.ndarray, face: object) -> bool:
        score = float(getattr(face, "det_score", 1.0))
        if score < settings.face_detection_threshold:
            return False
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, face.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if min(x2 - x1, y2 - y1) < settings.live_min_face_size:
            return False
        crop = frame[y1:y2, x1:x2]
        return crop.size > 0 and blur_score(crop) >= settings.live_blur_threshold

    @staticmethod
    def _face_belongs_to_person(
        face: object, person_boxes: list[tuple[int, int, int, int]],
    ) -> bool:
        x1, y1, x2, y2 = map(float, face.bbox)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        return any(
            px1 <= center_x <= px2 and py1 <= center_y <= py2
            for px1, py1, px2, py2 in person_boxes
        )

    @staticmethod
    def _draw(frame: np.ndarray, detections: list[DetectionView]) -> np.ndarray:
        output = frame.copy()
        for item in detections:
            x1, y1, x2, y2 = item.box
            known = item.employee_id is not None
            color = (46, 204, 113) if known else (60, 60, 230)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            identity = f"{item.employee_id} - {item.name}" if known else "UNKNOWN"
            lines = [identity, f"Similarity: {item.similarity:.2f}", item.attendance]
            top = max(22, y1 - 8 - 22 * (len(lines) - 1))
            for index, text in enumerate(lines):
                cv2.putText(output, text, (x1, top + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return output

    def latest(self) -> tuple[np.ndarray | None, list[DetectionView]]:
        with self._lock:
            detections = list(self._detections)
        ok, frame = self.camera.read()
        if not ok or frame is None:
            return None, detections
        # Draw cached recognition results over the newest capture frame. Display no
        # longer waits for ONNX inference, which removes most perceived latency.
        if self.activity is not None:
            try:
                frame = self.activity.draw(frame)
            except Exception as exc:
                logger.warning("Activity overlay skipped; attendance display continues: %s", exc)
        return self._draw(frame, detections), detections

    def activity_states(self) -> list[ActivityView]:
        return self.activity.latest() if self.activity is not None else []

    def _scale_for_display(
        self, detections: list[DetectionView], source_frame: np.ndarray,
    ) -> list[DetectionView]:
        if self.recognition_camera is self.camera:
            return detections
        ok, display_frame = self.camera.read()
        if not ok or display_frame is None:
            return detections
        source_height, source_width = source_frame.shape[:2]
        display_height, display_width = display_frame.shape[:2]
        if source_width == display_width and source_height == display_height:
            return detections
        scale_x, scale_y = display_width / source_width, display_height / source_height
        return [DetectionView(
            (
                int(item.box[0] * scale_x), int(item.box[1] * scale_y),
                int(item.box[2] * scale_x), int(item.box[3] * scale_y),
            ),
            item.employee_id, item.name, item.similarity, item.attendance,
        ) for item in detections]

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self.activity is not None:
            try:
                self.activity.stop()
            except Exception as exc:
                logger.warning("Activity stop failed: %s", exc)
        if self.recognition_camera is not self.camera:
            self.recognition_camera.stop()
        self.camera.stop()
