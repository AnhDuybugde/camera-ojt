from __future__ import annotations

from typing import Any

import numpy as np

from camera_tracking.domain import BoundingBox, Detection


def resolve_device(requested: str = "auto") -> str:
    """Chon device tot nhat: cuda neu co, khong thi mps, khong thi cpu."""
    name = (requested or "auto").strip().lower()
    if name not in ("auto", ""):
        if name.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    return name
            except ImportError:
                pass
            return "cpu"
        return name
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class YoloPersonDetector:
    """Ultralytics YOLO adapter; model loading is delayed until the first frame."""

    def __init__(
        self,
        model_path: str = "yolo26s.pt",
        confidence: float = 0.35,
        person_class_id: int = 0,
        image_size: int = 800,
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.person_class_id = person_class_id
        self.image_size = image_size
        self.device = resolve_device(device)
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError(
                    "Ultralytics is not installed. Run: pip install -r requirements.txt"
                ) from error
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.detect_batch([image])[0]

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        if not images:
            return []
        results = self._load_model().predict(
            source=images,
            classes=[self.person_class_id],
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            half=self.device.startswith("cuda"),
            verbose=False,
        )
        return [self._convert_result(result) for result in results]

    @staticmethod
    def _convert_result(result: Any) -> list[Detection]:
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for xyxy, score, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            detections.append(
                Detection(
                    bbox=BoundingBox(*(float(value) for value in xyxy)),
                    confidence=float(score),
                    class_id=int(class_id),
                )
            )
        return detections
