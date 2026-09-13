"""Runtime contracts shared by inference and downstream consumers."""

from camera_tracking.runtime.events import TrackEventBus
from camera_tracking.runtime.metrics import StageMetrics

__all__ = ["StageMetrics", "TrackEventBus"]
