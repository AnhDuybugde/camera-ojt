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
from face.liveness import LivenessDetector
from face.recognizer import FaceRecognizer

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
    ) -> None:
        self.camera, self.detector = camera, detector
        self.recognizer, self.attendance = recognizer, attendance
        self.liveness = LivenessDetector()
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
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="face-recognition")
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.camera.read()
            if not ok or frame is None:
                self.error = self.camera.error or "Waiting for camera..."
                time.sleep(0.1)
                continue
            started = time.monotonic()
            try:
                detections = self._recognize(frame)
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
        for face in self.detector.detect(frame):
            box = tuple(map(int, face.bbox))
            embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue
            match = self.recognizer.match(embedding)
            attendance_message = "UNKNOWN"
            if match.recognized:
                live = self.liveness.check(frame, face)
                if live.is_live:
                    attendance_message = self.attendance.record(match.employee_id).message  # type: ignore[arg-type]
                    logger.info("Employee recognized: %s score=%.3f", match.employee_id, match.similarity)
                else:
                    attendance_message = "LIVENESS FAILED"
            else:
                logger.debug("Unknown face score=%.3f", match.similarity)
            results.append(DetectionView(box, match.employee_id, match.full_name, match.similarity, attendance_message))
        return results

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
        return self._draw(frame, detections), detections

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self.camera.stop()
