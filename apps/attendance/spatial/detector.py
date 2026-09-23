"""Person detector with YOLO primary backend and OpenCV fallback."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PersonDetection:
    box: tuple[int, int, int, int]
    confidence: float


class PersonDetector:
    def __init__(self, model_path: str | Path, *, confidence: float = 0.35) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self._model = None
        self._hog = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self.backend_name = "Đang tải"

    def _load(self) -> None:
        if self._model is not None or self._hog is not None:
            return
        with self._load_lock:
            if self._model is not None or self._hog is not None:
                return
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(self.model_path))
                self.backend_name = "YOLO11s"
                logger.info("Spatial analytics loaded YOLO model: %s", self.model_path)
            except (ImportError, OSError, RuntimeError) as exc:
                logger.warning("YOLO unavailable; using OpenCV HOG fallback: %s", exc)
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self._hog = hog
                self.backend_name = "OpenCV HOG (dự phòng)"

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        if frame is None or frame.size == 0:
            return []
        self._load()
        with self._inference_lock:
            if self._model is not None:
                results = self._model.predict(
                    frame, classes=[0], conf=self.confidence, imgsz=640, verbose=False,
                )
                detections = []
                for result in results:
                    boxes = getattr(result, "boxes", None)
                    if boxes is None:
                        continue
                    for xyxy, score in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
                        detections.append(PersonDetection(tuple(map(int, xyxy)), float(score)))
                return detections

            height, width = frame.shape[:2]
            scale = min(1.0, 960.0 / max(width, height))
            working = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
            boxes, weights = self._hog.detectMultiScale(
                working, winStride=(8, 8), padding=(8, 8), scale=1.05,
            )
            detections = []
            for (x, y, w, h), score in zip(boxes, weights):
                if float(score) < self.confidence:
                    continue
                inv = 1.0 / scale
                detections.append(PersonDetection(
                    (int(x * inv), int(y * inv), int((x + w) * inv), int((y + h) * inv)),
                    float(score),
                ))
            return detections
