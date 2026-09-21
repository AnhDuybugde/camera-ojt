from __future__ import annotations

from camera_tracking.alerting import AlertDispatcher, AlertPublisher


def test_publisher_only_submits_enabled_attendance_events() -> None:
    dispatcher = AlertDispatcher("https://example.invalid/hook", "token")
    publisher = AlertPublisher(dispatcher, ["CHECK_IN"])
    assert publisher.publish(event_type="WAVE", occurred_at="2026-09-21T08:00:00+07:00", global_id=3, channel="A") is False
    assert publisher.publish(event_type="CHECK_IN", occurred_at="2026-09-21T08:00:00+07:00", global_id=3, channel="B", person_name="Van Dai") is True
    assert dispatcher.events.get_nowait() == {
        "event_type": "CHECK_IN",
        "person_name": "Van Dai",
        "occurred_at": "2026-09-21T08:00:00+07:00",
        "channel": "B",
    }
