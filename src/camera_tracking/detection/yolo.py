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
        nms_iou_threshold: float = 0.50,
        nested_box_containment_threshold: float = 0.85,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.person_class_id = person_class_id
        self.image_size = image_size
        self.device = resolve_device(device)
        self.nms_iou_threshold = max(0.0, min(1.0, nms_iou_threshold))
        self.nested_box_containment_threshold = max(
            0.0, min(1.0, nested_box_containment_threshold)
        )
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
            iou=self.nms_iou_threshold,
            device=self.device,
            half=self.device.startswith("cuda"),
            verbose=False,
        )
        return [
            suppress_nested_detections(
                self._convert_result(result),
                containment_threshold=self.nested_box_containment_threshold,
            )
            for result in results
        ]

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


def suppress_nested_detections(
    detections: list[Detection], *, containment_threshold: float = 0.85
) -> list[Detection]:
    """Remove a detection almost fully contained by a stronger detection.

    Ultralytics NMS compares intersection over *union*.  A small duplicate
    person box inside a larger box can therefore survive NMS even when the
    small box is completely covered.  This second, conservative pass compares
    intersection over the smaller box area and only removes near-containment.
    It does not suppress two ordinary side-by-side people.
    """
    if len(detections) < 2:
        return detections
    threshold = max(0.0, min(1.0, containment_threshold))
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence,
                            reverse=True):
        candidate_area = candidate.bbox.area
        duplicate = False
        for stronger in kept:
            intersection = _intersection_area(candidate, stronger)
            if candidate_area <= 0 or intersection / candidate_area < threshold:
                continue
            duplicate = True
            break
        if not duplicate:
            kept.append(candidate)
    return kept


def _intersection_area(left: Detection, right: Detection) -> float:
    x1 = max(left.bbox.x1, right.bbox.x1)
    y1 = max(left.bbox.y1, right.bbox.y1)
    x2 = min(left.bbox.x2, right.bbox.x2)
    y2 = min(left.bbox.y2, right.bbox.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
