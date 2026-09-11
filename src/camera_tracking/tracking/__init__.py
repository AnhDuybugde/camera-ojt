"""Multi-object tracking modules."""
from camera_tracking.tracking.byte import ByteTrackTracker
from camera_tracking.tracking.global_identity import (
    GlobalIdentityConfig,
    GlobalIdentityManager,
    IdentityRecord,
    IdentityState,
)
from camera_tracking.tracking.identity import PersistentIdentityTracker
from camera_tracking.tracking.iou import IoUTracker
from camera_tracking.tracking.stability import StablePersonCount

__all__ = [
    "ByteTrackTracker",
    "GlobalIdentityConfig",
    "GlobalIdentityManager",
    "IdentityRecord",
    "IdentityState",
    "IoUTracker",
    "PersistentIdentityTracker",
    "StablePersonCount",
]
