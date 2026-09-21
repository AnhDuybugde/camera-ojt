from datetime import datetime, time, timedelta

from attendance.attendance_rules import can_check_out, check_in_status, work_session_at
from config import settings


def test_on_time_and_late() -> None:
    threshold = time(9, 15)
    assert check_in_status(datetime(2026, 9, 14, 9, 15), threshold) == "ON_TIME"
    assert check_in_status(datetime(2026, 9, 14, 9, 16), threshold) == "LATE"


def test_checkout_minimum_interval() -> None:
    now = datetime(2026, 9, 14, 12, 0)
    assert can_check_out((now - timedelta(minutes=240)).isoformat(), now, 240)
    assert not can_check_out((now - timedelta(minutes=239)).isoformat(), now, 240)


def test_work_sessions_exclude_lunch_break() -> None:
    assert work_session_at(datetime(2026, 9, 14, 11, 59)) == "MORNING"
    assert work_session_at(datetime(2026, 9, 14, 12, 0)) is None
    assert work_session_at(datetime(2026, 9, 14, 13, 59)) is None
    assert work_session_at(datetime(2026, 9, 14, 14, 0)) == "AFTERNOON"


def test_official_work_time_configuration() -> None:
    assert settings.work_start_time == time(9, 0)
    assert settings.morning_end_time == time(12, 0)
    assert settings.afternoon_start_time == time(14, 0)
    assert settings.work_end_time == time(17, 30)
    assert settings.late_threshold == time(9, 15)
