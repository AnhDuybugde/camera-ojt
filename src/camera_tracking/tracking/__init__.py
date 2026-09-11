"""Multi-object tracking modules."""
from camera_tracking.tracking.byte import ByteTrackTracker
from camera_tracking.tracking.identity import PersistentIdentityTracker
from camera_tracking.tracking.iou import IoUTracker
from camera_tracking.tracking.stability import StablePersonCount

__all__ = [
    "ByteTrackTracker",
    "IoUTracker",
    "PersistentIdentityTracker",
    "StablePersonCount",
]
