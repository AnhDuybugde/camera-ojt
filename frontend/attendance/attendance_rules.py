"""Pure attendance rules, kept separately for easy testing."""
from __future__ import annotations

from datetime import datetime, time


def check_in_status(at: datetime, late_threshold: time) -> str:
    return "ON_TIME" if at.time().replace(tzinfo=None) <= late_threshold else "LATE"


def can_check_out(check_in_iso: str, now: datetime, minimum_minutes: int) -> bool:
    checked_in = datetime.fromisoformat(check_in_iso)
    if checked_in.tzinfo is None and now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    return (now - checked_in).total_seconds() >= minimum_minutes * 60


def work_session_at(
    at: datetime,
    morning_end: time = time(12, 0),
    afternoon_start: time = time(14, 0),
) -> str | None:
    """Return the active session, or None during the configured lunch break."""
    current = at.time().replace(tzinfo=None)
    if current < morning_end:
        return "MORNING"
    if current < afternoon_start:
        return None
    return "AFTERNOON"


def scheduled_absent_ids(
    scheduled_on_ids: set[str], present_ids: set[str]
) -> set[str]:
    """Only employees scheduled ON and missing a check-in are absent."""
    return scheduled_on_ids - present_ids
