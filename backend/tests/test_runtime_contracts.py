import time

import numpy as np

from camera_tracking.camera import FrameHub
from camera_tracking.domain import BoundingBox, Frame, Track, TrackEvent
from camera_tracking.runtime import StageMetrics, TrackEventBus
from camera_tracking.streaming import LatestJpegRenderer


def test_frame_hub_keeps_latest_frame_only() -> None:
    hub = FrameHub()
    first = Frame(1, 1.0, np.zeros((2, 2, 3), dtype=np.uint8))
    second = Frame(2, 2.0, np.ones((2, 2, 3), dtype=np.uint8))
    hub.publish("A", first)
    hub.publish("A", second)

    assert hub.latest("A") is second
    assert hub.latest("B") is None


def test_track_event_bus_fans_out_same_event() -> None:
    frame = Frame(1, 1.0, np.zeros((2, 2, 3), dtype=np.uint8))
    track = Track(1, BoundingBox(0, 0, 1, 1), 0.9, 1, 1, True)
    event = TrackEvent("B", frame, (track,))
    received: list[TrackEvent] = []
    bus = TrackEventBus()
    bus.subscribe(received.append)
    bus.subscribe(received.append)

    bus.publish(event)

    assert received == [event]


def test_stage_metrics_records_measurements() -> None:
    metrics = StageMetrics()

    with metrics.measure("detection"):
        pass

    snapshot = metrics.snapshot()
    assert snapshot["detection"]["count"] == 1.0
    assert snapshot["detection"]["mean_ms"] >= 0.0


def test_latest_jpeg_renderer_publishes_latest_frame() -> None:
    class FakeStreamer:
        def __init__(self) -> None:
            self.published: list[tuple[str, bytes]] = []

        def push(self, name: str, payload: bytes) -> None:
            self.published.append((name, payload))

    streamer = FakeStreamer()
    renderer = LatestJpegRenderer(streamer)  # type: ignore[arg-type]
    renderer.start()
    renderer.submit("A", np.zeros((4, 4, 3), dtype=np.uint8))

    for _ in range(20):
        if streamer.published:
            break
        time.sleep(0.01)
    renderer.stop()

    assert streamer.published
    assert streamer.published[0][0] == "A"
    assert streamer.published[0][1][:2] == b"\xff\xd8"
