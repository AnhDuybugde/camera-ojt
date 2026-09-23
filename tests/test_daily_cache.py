"""Test DailyStateCache: attendance 1 lan/ngay, in_room throttle 1 gio."""
from camera_tracking.store.daily import DailyStateCache


def test_attendance_write_once() -> None:
    cache = DailyStateCache()
    assert cache.attendance_should_write("2026-09-12", "An") is True
    assert cache.attendance_should_write("2026-09-12", "An") is False
    assert cache.attendance_should_write("2026-09-12", "Bo") is True
    assert cache.attendance_should_write("2026-09-13", "An") is True


def test_status_writes_on_label_change_only() -> None:
    cache = DailyStateCache(inroom_min_interval_s=3600.0)
    write, log = cache.status_should_write(
        day="2026-09-12", global_id=1, label="Working", in_room=True, now_s=0.0)
    assert (write, log) == (True, True)
    # Khong doi -> khong ghi.
    write, log = cache.status_should_write(
        day="2026-09-12", global_id=1, label="Working", in_room=True, now_s=10.0)
    assert (write, log) == (False, False)
    # Doi label (khong flip in_room) -> ghi.
    write, _log = cache.status_should_write(
        day="2026-09-12", global_id=1, label="Away", in_room=True, now_s=20.0)
    assert write is True


def test_status_heartbeat_is_bounded_to_one_write_per_interval() -> None:
    cache = DailyStateCache(heartbeat_interval_s=30.0)
    assert cache.status_should_write(day="2026-09-12", global_id=8,
        label="Working", in_room=True, now_s=0.0) == (True, True)
    assert cache.status_should_write(day="2026-09-12", global_id=8,
        label="Working", in_room=True, now_s=29.9) == (False, False)
    assert cache.status_should_write(day="2026-09-12", global_id=8,
        label="Working", in_room=True, now_s=30.0) == (True, False)


def test_inroom_flip_throttled_one_hour() -> None:
    cache = DailyStateCache(inroom_min_interval_s=3600.0)
    cache.status_should_write(day="2026-09-12", global_id=2,
                              label="Working", in_room=True, now_s=0.0)
    # Flip sau 60s (kem label doi) -> bi chan row-write, chi log event.
    write, log = cache.status_should_write(day="2026-09-12", global_id=2,
                                           label="Out of office",
                                           in_room=False, now_s=60.0)
    assert write is False and log is True
    # Flip sau >1h -> cho ghi.
    write, _log = cache.status_should_write(day="2026-09-12", global_id=2,
                                            label="Out of office",
                                            in_room=False, now_s=3700.0)
    assert write is True
