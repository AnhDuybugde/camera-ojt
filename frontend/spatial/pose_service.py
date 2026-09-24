"""Reusable YOLO Pose service. Loading/inference are serialized across cameras."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from spatial.activity_config import ActivityConfig

logger = logging.getLogger(__name__)

COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)


@dataclass(frozen=True, slots=True)
class PoseObservation:
    box: tuple[int, int, int, int]
    confidence: float
    keypoints: dict[str, tuple[float, float, float]]


class PoseService:
    """Lazily loads one pose model and returns person boxes plus COCO keypoints."""

    def __init__(self, config: ActivityConfig) -> None:
        self.config = config
        self._model = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self.backend_name = "YOLO Pose (chưa tải)"

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from ultralytics import YOLO
            self._model = YOLO(self.config.pose_model)
            self.backend_name = f"YOLO Pose · {self.config.pose_model}"
            logger.info("Loaded activity pose model: %s", self.config.pose_model)

    def infer(self, frame: np.ndarray) -> list[PoseObservation]:
        if frame is None or frame.size == 0:
            return []
        self._load()
        with self._inference_lock:
            results = self._model.predict(  # type: ignore[union-attr]
                frame, classes=[0], conf=self.config.pose_confidence,
                imgsz=self.config.pose_image_size, verbose=False,
            )
        observations: list[PoseObservation] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            poses = getattr(result, "keypoints", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            keypoint_data = None if poses is None or poses.data is None else poses.data.cpu().numpy()
            for index, (box, score) in enumerate(zip(xyxy, scores)):
                points: dict[str, tuple[float, float, float]] = {}
                if keypoint_data is not None and index < len(keypoint_data):
                    for name, point in zip(COCO_KEYPOINTS, keypoint_data[index]):
                        confidence = float(point[2]) if len(point) > 2 else 1.0
                        points[name] = (float(point[0]), float(point[1]), confidence)
                observations.append(PoseObservation(tuple(map(int, box)), float(score), points))
        return observations
