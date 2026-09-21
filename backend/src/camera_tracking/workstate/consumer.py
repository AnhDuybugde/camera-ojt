from __future__ import annotations

from camera_tracking.domain import Point, TrackEvent
from camera_tracking.workstate.channel_status import ChannelBusinessTracker, PersonBusinessState


class WorkstateConsumer:
    """Adapt immutable track events to the channel business state machine."""

    def __init__(self, tracker: ChannelBusinessTracker) -> None:
        self.tracker = tracker

    def consume(
        self,
        event: TrackEvent,
        positions: dict[int, Point] | None = None,
        persons: dict[int, str] | None = None,
    ) -> dict[int, PersonBusinessState]:
        global_ids = {
            track.global_person_id
            if track.global_person_id is not None
            else track.track_id
            for track in event.tracks
        }
        return self.tracker.update(
            event.frame.timestamp_s,
            global_ids,
            positions,
            persons,
        )


__all__ = ["WorkstateConsumer"]
