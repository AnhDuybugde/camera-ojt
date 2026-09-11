"""ROI theo pixel ảnh cho từng vị trí ngồi (channel A) và vùng channel B."""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from camera_tracking.domain import BoundingBox, Point, Track


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Ray-casting, không phụ thuộc OpenCV để dễ unit-test."""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def bbox_center_in_polygon(bbox: BoundingBox, polygon: list[Point]) -> bool:
    cx = (bbox.x1 + bbox.x2) / 2.0
    cy = (bbox.y1 + bbox.y2) / 2.0
    return point_in_polygon((cx, cy), polygon)


@dataclass(slots=True)
class SeatZone:
    """Một vị trí ngồi trong channel A (tọa độ pixel ảnh)."""

    seat_id: str
    name: str
    polygon: list[Point]
    channel: str = "A"

    def contains(self, bbox: BoundingBox) -> bool:
        return bbox_center_in_polygon(bbox, self.polygon)


def select_seat_occupant(
    seat: SeatZone,
    tracks: list[Track],
    *,
    preferred_track_id: int | None = None,
    ambiguity_margin: float = 0.12,
) -> tuple[Track | None, bool]:
    """Select the track closest to the seat-facing edge of a desk ROI."""
    candidates = [track for track in tracks if seat.contains(track.bbox)]
    if not candidates:
        return None, False

    if preferred_track_id is not None:
        preferred = next(
            (track for track in candidates if track.track_id == preferred_track_id),
            None,
        )
        if preferred is not None:
            return preferred, False

    xs = [point[0] for point in seat.polygon]
    ys = [point[1] for point in seat.polygon]
    width = max(1.0, max(xs) - min(xs))
    height = max(1.0, max(ys) - min(ys))
    anchor_x = (min(xs) + max(xs)) / 2.0
    anchor_y = max(ys)

    def distance(track: Track) -> float:
        foot_x = (track.bbox.x1 + track.bbox.x2) / 2.0
        foot_y = track.bbox.y2
        return hypot((foot_x - anchor_x) / width, (foot_y - anchor_y) / height)

    ranked = sorted(candidates, key=distance)
    if len(ranked) > 1 and distance(ranked[1]) - distance(ranked[0]) < ambiguity_margin:
        return None, True
    return ranked[0], False


@dataclass(slots=True)
class CorridorZones:
    """Các vùng logic của channel B."""

    hallway: list[Point]
    exit_door: list[Point]

    def locate(self, bbox: BoundingBox) -> str | None:
        """Trả về 'exit' | 'hallway' | None."""
        if bbox_center_in_polygon(bbox, self.exit_door):
            return "exit"
        if bbox_center_in_polygon(bbox, self.hallway):
            return "hallway"
        return None
