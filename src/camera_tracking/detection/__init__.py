"""Person detection modules."""
from camera_tracking.detection.base import PersonDetector
from camera_tracking.detection.yolo import YoloPersonDetector

__all__ = ["PersonDetector", "YoloPersonDetector"]
