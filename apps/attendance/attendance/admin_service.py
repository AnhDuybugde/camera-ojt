"""Compatibility shim — canonical home is backend.app.services.attendance_admin_service."""
from backend.app.services.attendance_admin_service import (
    ATTENDANCE_STATUSES,
    AUDITED_FIELDS,
    AttendanceAdminService,
)

__all__ = ["ATTENDANCE_STATUSES", "AUDITED_FIELDS", "AttendanceAdminService"]
