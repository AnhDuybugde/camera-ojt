"""Compatibility shim — canonical home is frontend.src.hooks.common."""
from frontend.src.hooks.common import (
    get_attendance_admin_service,
    get_attendance_service,
    get_auth_service,
    get_db,
    get_detector,
    get_recognizer,
    get_sync_service,
    get_sync_worker,
)

__all__ = [
    "get_attendance_admin_service",
    "get_attendance_service",
    "get_auth_service",
    "get_db",
    "get_detector",
    "get_recognizer",
    "get_sync_service",
    "get_sync_worker",
]
