from types import SimpleNamespace

from camera_tracking.streaming import mjpeg
from camera_tracking.store.queue import WriteQueue
from camera_tracking.workstate.channel_status import ChannelBusinessTracker, PersonBusinessState


def test_ui_status_is_latest_value_and_at_most_two_updates_per_second(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(mjpeg.time, "monotonic", lambda: now[0])
    server = mjpeg.MjpegStreamer()
    server.set_status({"value": 0})
    server.set_status({"value": 1})
    server.set_status({"value": 2})
    assert server.snapshot()[1]["value"] == 0
    now[0] = 10.5
    assert server.snapshot()[1]["value"] == 2


def test_queue_ack_does_not_drop_coalesced_update(tmp_path):
    queue = WriteQueue(tmp_path / "queue.db")
    row_id = queue.push("room_status", {"date": "2026-09-23", "global_id": 4,
                                         "label": "Working"})
    old = queue.peek()[0][2]
    queue.push("room_status", {"date": "2026-09-23", "global_id": 4,
                                "label": "Away"})
    queue.ack(row_id, old)
    assert queue.peek()[0][2]["label"] == "Away"


def test_no_zone_local_presence_defaults_to_working():
    tracker = ChannelBusinessTracker(channel="A", zone_mode="none", move_ratio=0)
    state = tracker.update(1.0, {12})
    assert state[12] is PersonBusinessState.WORKING
