"""Attendance lifecycle primitives used by the production camera runtime."""

from .lifecycle import (
    AttendanceLifecycleEngine,
    AttendanceLifecycleEvent,
    AttendanceLifecycleRecord,
    AttendanceState,
)

__all__ = [
    "AttendanceLifecycleEngine",
    "AttendanceLifecycleEvent",
    "AttendanceLifecycleRecord",
    "AttendanceState",
]
