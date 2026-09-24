from camera_tracking.attendance import (
    AttendanceLifecycleEngine,
    AttendanceLifecycleRecord,
    AttendanceState,
)


def test_first_in_last_out_and_return() -> None:
    engine = AttendanceLifecycleEngine()
    events = engine.observe_enter(
        day="2026-09-24", person_id="E01", display_name="Minh", global_id=7,
        at_iso="2026-09-24T08:00:00+07:00", channel="B", confidence=0.92,
    )
    assert [e.event for e in events] == ["CHECK_IN"]
    record = engine.record_of("2026-09-24", "E01")
    assert record is not None and record.state == AttendanceState.PRESENT

    events = engine.observe_exit(
        day="2026-09-24", person_id="E01", display_name="Minh", global_id=7,
        at_iso="2026-09-24T12:00:00+07:00", channel="B", confidence=0.91,
    )
    assert [e.event for e in events] == ["TEMP_OUT"]
    assert record.check_out_at.endswith("12:00:00+07:00")

    events = engine.observe_enter(
        day="2026-09-24", person_id="E01", display_name="Minh", global_id=8,
        at_iso="2026-09-24T13:00:00+07:00", channel="B", confidence=0.93,
    )
    assert [e.event for e in events] == ["RETURN"]
    assert record.state == AttendanceState.PRESENT
    assert record.check_out_at is None

    engine.observe_exit(
        day="2026-09-24", person_id="E01", display_name="Minh", global_id=8,
        at_iso="2026-09-24T17:30:00+07:00", channel="B", confidence=0.93,
    )
    events = engine.finalize_checkout(day="2026-09-24", person_id="E01")
    assert [e.event for e in events] == ["CHECK_OUT"]
    assert record.state == AttendanceState.CHECKED_OUT
    assert record.check_out_at.endswith("17:30:00+07:00")


def test_exit_cannot_invent_checkin() -> None:
    engine = AttendanceLifecycleEngine()
    events = engine.observe_exit(
        day="2026-09-24", person_id="E02", display_name="Lan", global_id=4,
        at_iso="2026-09-24T10:00:00+07:00", channel="B",
    )
    assert events == []
    record = engine.record_of("2026-09-24", "E02")
    assert record is not None
    assert record.check_in_at is None


def test_preload_and_repeated_exit_do_not_duplicate_events() -> None:
    engine = AttendanceLifecycleEngine()
    engine.preload([AttendanceLifecycleRecord(
        day="2026-09-24",
        person_id="E03",
        display_name="An",
        state=AttendanceState.PRESENT,
        check_in_at="2026-09-24T08:00:00+07:00",
        global_id=3,
    )])
    first = engine.observe_exit(
        day="2026-09-24", person_id="E03", display_name="An", global_id=3,
        at_iso="2026-09-24T12:00:00+07:00", channel="B",
    )
    repeated = engine.observe_exit(
        day="2026-09-24", person_id="E03", display_name="An", global_id=3,
        at_iso="2026-09-24T12:00:05+07:00", channel="B",
    )
    assert [event.event for event in first] == ["TEMP_OUT"]
    assert repeated == []
    record = engine.record_of("2026-09-24", "E03")
    assert record is not None
    assert record.check_out_at == "2026-09-24T12:00:05+07:00"
