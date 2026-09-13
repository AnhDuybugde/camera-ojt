import numpy as np

from camera_tracking.domain import BoundingBox, Frame, Track, TrackEvent
from camera_tracking.workstate import ChannelBusinessTracker, PersonBusinessState, WorkstateConsumer


def test_workstate_consumer_uses_global_identity_from_event() -> None:
    tracker = ChannelBusinessTracker(
        channel="A", away_grace_s=10.0, out_after_s=60.0, return_stable_s=3.0
    )
    consumer = WorkstateConsumer(tracker)
    track = Track(
        99, BoundingBox(0, 0, 10, 20), 0.9, 1, 3, True,
        local_track_id=4, global_person_id=17,
    )
    event = TrackEvent("A", Frame(1, 5.0, np.zeros((20, 20, 3), dtype=np.uint8)), (track,))

    states = consumer.consume(event)

    assert states[17] is PersonBusinessState.WORKING
    assert 99 not in states
