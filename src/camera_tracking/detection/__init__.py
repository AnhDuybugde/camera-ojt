"""Person detection modules."""
from camera_tracking.detection.base import PersonDetector
from camera_tracking.detection.yolo import YoloPersonDetector, resolve_device

__all__ = ["PersonDetector", "YoloPersonDetector", "resolve_device"]
