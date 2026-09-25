"""Multi-object tracking modules."""
from backend.app.recognition.tracker import ByteTrackTracker
from camera_tracking.tracking.global_identity import (
    GlobalIdentityConfig,
    GlobalIdentityManager,
    IdentityRecord,
    IdentityState,
)
from camera_tracking.tracking.identity import PersistentIdentityTracker
from camera_tracking.tracking.iou import IoUTracker
from camera_tracking.tracking.stability import StablePersonCount
from camera_tracking.tracking.state_store import DailyIdentityStore

__all__ = [
    "ByteTrackTracker",
    "DailyIdentityStore",
    "GlobalIdentityConfig",
    "GlobalIdentityManager",
    "IdentityRecord",
    "IdentityState",
    "IoUTracker",
    "PersistentIdentityTracker",
    "StablePersonCount",
]
