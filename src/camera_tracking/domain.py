from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def foot_point(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, self.y2)


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: int = 0


@dataclass(frozen=True, slots=True)
class Track:
    track_id: int
    bbox: BoundingBox
    confidence: float
    age: int
    hits: int
    confirmed: bool


@dataclass(frozen=True, slots=True)
class Frame:
    index: int
    timestamp_s: float
    image: np.ndarray


@dataclass(slots=True)
class AnalyticsSnapshot:
    frame_index: int
    timestamp_s: float
    occupancy: int
    entries: int
    exits: int
    density_grid: list[list[int]]
    heatmap_grid: list[list[int]]
    zone_counts: dict[str, int]
    floor_positions: dict[int, Point]
    movement_vectors: dict[int, Point]
    trajectories: dict[int, list[Point]] = field(default_factory=dict)
