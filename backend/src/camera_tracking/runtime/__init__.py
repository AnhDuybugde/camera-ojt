"""Runtime contracts shared by inference and downstream consumers."""

from camera_tracking.runtime.events import TrackEventBus
from camera_tracking.runtime.metrics import StageMetrics
from camera_tracking.runtime.presence import PresenceSummary, summarize_presence

__all__ = [
    "PresenceSummary",
    "StageMetrics",
    "TrackEventBus",
    "summarize_presence",
]
