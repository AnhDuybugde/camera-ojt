"""Camera AIM audio + event-driven Bé Xinh companion."""

from camera_tracking.audio.announcer import CameraCheckInAnnouncer
from camera_tracking.audio.companion import HamyCompanion
from camera_tracking.audio.gesture import GestureEvent, HandGestureDetector
from camera_tracking.audio.interaction import InteractionEngine, InteractionEvent, MotionSnapshot

__all__ = [
    "CameraCheckInAnnouncer",
    "HamyCompanion",
    "GestureEvent",
    "HandGestureDetector",
    "InteractionEngine",
    "InteractionEvent",
    "MotionSnapshot",
]
