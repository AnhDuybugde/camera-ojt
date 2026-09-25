"""Compatibility shim — canonical home is backend.app.services.attendance_service."""
from backend.app.services.attendance_service import AttendanceResult, AttendanceService

__all__ = ["AttendanceResult", "AttendanceService"]
