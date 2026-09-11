"""ROI theo pixel ảnh cho từng vị trí ngồi (channel A) và vùng channel B."""
from __future__ import annotations

from dataclasses import dataclass

from camera_tracking.domain import BoundingBox, Point


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
