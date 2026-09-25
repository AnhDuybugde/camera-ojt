"""Compatibility shim — canonical home is backend.app.services.attendance_rules."""
from backend.app.services.attendance_rules import (
    can_check_out,
    check_in_status,
    scheduled_absent_ids,
    work_session_at,
)

__all__ = ["can_check_out", "check_in_status", "scheduled_absent_ids", "work_session_at"]
