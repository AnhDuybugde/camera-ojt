from __future__ import annotations

from collections.abc import Callable

from camera_tracking.domain import TrackEvent

TrackConsumer = Callable[[TrackEvent], None]


class TrackEventBus:
    """Small synchronous fan-out bus for one inference tick."""

    def __init__(self) -> None:
        self._consumers: list[TrackConsumer] = []

    def subscribe(self, consumer: TrackConsumer) -> None:
        if consumer not in self._consumers:
            self._consumers.append(consumer)

    def publish(self, event: TrackEvent) -> None:
        for consumer in tuple(self._consumers):
            consumer(event)
