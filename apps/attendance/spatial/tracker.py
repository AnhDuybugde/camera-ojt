"""Small dependency-free centroid tracker for stable person IDs."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


Box = tuple[int, int, int, int]


def _anchor(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


@dataclass(slots=True)
class Track:
    track_id: int
    box: Box
    missed: int = 0
    employee_id: str | None = None
    employee_name: str = "Chưa xác định"
    trail: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=24))

    def __post_init__(self) -> None:
        self.trail.append(_anchor(self.box))


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    track_id: int
    box: Box
    anchor: tuple[float, float]
    employee_id: str | None
    employee_name: str
    trail: tuple[tuple[float, float], ...]


class CentroidTracker:
    def __init__(self, *, max_distance: float = 140.0, max_missed: int = 5) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, Track] = {}
        self._retired: list[TrackSnapshot] = []

    def update(self, boxes: list[Box]) -> list[TrackSnapshot]:
        unmatched_tracks = set(self._tracks)
        unmatched_boxes = set(range(len(boxes)))
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            tx, ty = _anchor(track.box)
            for box_index, box in enumerate(boxes):
                bx, by = _anchor(box)
                candidates.append((math.hypot(tx - bx, ty - by), track_id, box_index))

        matched_ids: set[int] = set()
        for distance, track_id, box_index in sorted(candidates):
            if distance > self.max_distance:
                break
            if track_id not in unmatched_tracks or box_index not in unmatched_boxes:
                continue
            track = self._tracks[track_id]
            track.box = boxes[box_index]
            track.missed = 0
            track.trail.append(_anchor(track.box))
            matched_ids.add(track_id)
            unmatched_tracks.remove(track_id)
            unmatched_boxes.remove(box_index)

        for track_id in unmatched_tracks:
            self._tracks[track_id].missed += 1
        for track_id in list(unmatched_tracks):
            track = self._tracks[track_id]
            if track.missed > self.max_missed:
                self._retired.append(self._snapshot(track))
                del self._tracks[track_id]

        for box_index in sorted(unmatched_boxes):
            track = Track(self._next_id, boxes[box_index])
            self._tracks[track.track_id] = track
            matched_ids.add(track.track_id)
            self._next_id += 1

        return [self._snapshot(self._tracks[track_id]) for track_id in sorted(matched_ids)]

    def identify(self, track_id: int, employee_id: str, employee_name: str) -> None:
        track = self._tracks.get(track_id)
        if track:
            track.employee_id = employee_id
            track.employee_name = employee_name

    def snapshot(self, track_id: int) -> TrackSnapshot | None:
        track = self._tracks.get(track_id)
        return self._snapshot(track) if track else None

    def pop_retired(self) -> list[TrackSnapshot]:
        retired, self._retired = self._retired, []
        return retired

    @staticmethod
    def _snapshot(track: Track) -> TrackSnapshot:
        return TrackSnapshot(
            track.track_id, track.box, _anchor(track.box), track.employee_id,
            track.employee_name, tuple(track.trail),
        )
