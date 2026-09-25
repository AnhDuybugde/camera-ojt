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
                from pathlib import Path
                from camera_tracking.config import load_config
                root = Path(__file__).resolve().parents[3]
                cfg = load_config(root / "config/default.yaml").face
                logger.info("Loading configured enrollment model: %s", cfg.model_pack)
                model = FaceAnalysis(name=cfg.model_pack, providers=["CPUExecutionProvider"],
                                     allowed_modules=["detection", "recognition"])
                model.prepare(ctx_id=-1, det_size=(cfg.det_size, cfg.det_size))
                self._model = model
        return self._model

    def detect(self, frame: np.ndarray) -> list[Any]:
        if frame is None or frame.size == 0:
            return []
        with self._inference_lock:
            return list(self._load().get(frame))
