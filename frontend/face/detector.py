"""Lazy InsightFace model loader and face detector."""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self, detection_size: int | None = None) -> None:
        self.detection_size = detection_size or settings.detection_size
        self._model: Any = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from insightface.app import FaceAnalysis
                except ImportError as exc:
                    raise RuntimeError(
                        "InsightFace is not installed. Run: pip install -r requirements.txt"
                    ) from exc
                logger.info("Loading InsightFace buffalo_l model")
                # Live attendance only needs the detector and ArcFace embedding.
                # Skipping age/gender and dense landmark models reduces CPU load
                # without changing the face recognition model.
                model = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection", "recognition"],
                    providers=["CPUExecutionProvider"],
                )
                model.prepare(
                    ctx_id=-1,
                    det_thresh=settings.face_detection_threshold,
                    det_size=(self.detection_size, self.detection_size),
                )
                self._model = model
        return self._model

    def detect(self, frame: np.ndarray) -> list[Any]:
        if frame is None or frame.size == 0:
            return []
        with self._inference_lock:
            return list(self._load().get(frame))
