"""Normalized camera zones used by the spatial analytics pipeline."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Zone:
    zone_id: str
    label: str
    points: tuple[tuple[float, float], ...]
    color: tuple[int, int, int]

    def contains(self, x: float, y: float) -> bool:
        """Return True when a normalized point lies inside the polygon."""
        inside = False
        previous = self.points[-1]
        for current in self.points:
            x1, y1 = previous
            x2, y2 = current
            if (y1 > y) != (y2 > y):
                crossing_x = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
                if x < crossing_x:
                    inside = not inside
            previous = current
        return inside


DEFAULT_ZONES = (
    Zone("entrance", "Lối vào", ((0.00, 0.00), (0.32, 0.00), (0.32, 1.00), (0.00, 1.00)), (235, 158, 52)),
    Zone("center", "Khu trung tâm", ((0.32, 0.00), (0.68, 0.00), (0.68, 1.00), (0.32, 1.00)), (79, 70, 229)),
    Zone("workspace", "Khu làm việc", ((0.68, 0.00), (1.00, 0.00), (1.00, 1.00), (0.68, 1.00)), (16, 185, 129)),
)


def zones_from_environment() -> tuple[Zone, ...]:
    """Load zones from SPATIAL_ZONES_JSON, falling back to safe demo zones."""
    raw = os.getenv("SPATIAL_ZONES_JSON", "").strip()
    if not raw:
        return DEFAULT_ZONES
    try:
        payload = json.loads(raw)
        colors = ((235, 158, 52), (79, 70, 229), (16, 185, 129), (220, 38, 38))
        zones = []
        for index, item in enumerate(payload):
            points = tuple((float(x), float(y)) for x, y in item["points"])
            if len(points) < 3 or any(not 0 <= value <= 1 for point in points for value in point):
                raise ValueError("Zone coordinates must be normalized polygons")
            zones.append(Zone(str(item["id"]), str(item["label"]), points, colors[index % len(colors)]))
        return tuple(zones) or DEFAULT_ZONES
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return DEFAULT_ZONES


def zone_at(zones: Iterable[Zone], x: float, y: float) -> Zone | None:
    return next((zone for zone in zones if zone.contains(x, y)), None)
