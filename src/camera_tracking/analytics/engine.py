from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np

from camera_tracking.analytics.projection import FloorProjector
from camera_tracking.domain import AnalyticsSnapshot, Point, Track


@dataclass(frozen=True, slots=True)
class Zone:
    name: str
    points: list[Point]


@dataclass(frozen=True, slots=True)
class CountingLine:
    start: Point
    end: Point
    entry_direction: str = "negative_to_positive"


def _side_of_line(point: Point, line: CountingLine) -> float:
    return (line.end[0] - line.start[0]) * (point[1] - line.start[1]) - (
        line.end[1] - line.start[1]
    ) * (point[0] - line.start[0])


def _segments_cross(
    first_start: Point,
    first_end: Point,
    second_start: Point,
    second_end: Point,
) -> bool:
    def orientation(start: Point, end: Point, point: Point) -> float:
        return (end[0] - start[0]) * (point[1] - start[1]) - (
            end[1] - start[1]
        ) * (point[0] - start[0])

    return orientation(first_start, first_end, second_start) * orientation(
        first_start, first_end, second_end
    ) <= 0 and orientation(second_start, second_end, first_start) * orientation(
        second_start, second_end, first_end
    ) <= 0


class RoomAnalytics:
    def __init__(
        self,
        projector: FloorProjector,
        floor_width_m: float,
        floor_height_m: float,
        grid_rows: int,
        grid_cols: int,
        zones: list[Zone] | None = None,
        counting_line: CountingLine | None = None,
        trajectory_length: int = 120,
    ) -> None:
        self.projector = projector
        self.floor_width_m = floor_width_m
        self.floor_height_m = floor_height_m
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.zones = zones or []
        self.counting_line = counting_line
        self.trajectories: dict[int, deque[Point]] = defaultdict(
            lambda: deque(maxlen=trajectory_length)
        )
        self._line_sides: dict[int, float] = {}
        self._line_positions: dict[int, Point] = {}
        self._heatmap = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]
        self.entries = 0
        self.exits = 0

    def update(
        self, frame_index: int, timestamp_s: float, tracks: list[Track]
    ) -> AnalyticsSnapshot:
        confirmed_tracks = [track for track in tracks if track.confirmed]
        positions: dict[int, Point] = {}
        density = [[0 for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        zone_counts = {zone.name: 0 for zone in self.zones}
        movement_vectors: dict[int, Point] = {}

        for track in confirmed_tracks:
            position = self.projector.project(track.bbox.foot_point)
            positions[track.track_id] = position
            previous_position = (
                self.trajectories[track.track_id][-1]
                if self.trajectories[track.track_id]
                else position
            )
            movement_vectors[track.track_id] = (
                position[0] - previous_position[0],
                position[1] - previous_position[1],
            )
            self.trajectories[track.track_id].append(position)
            self._accumulate_density(density, position)
            self._accumulate_density(self._heatmap, position)
            for zone in self.zones:
                polygon = np.asarray(zone.points, dtype=np.float32)
                if len(polygon) >= 3 and cv2.pointPolygonTest(polygon, position, False) >= 0:
                    zone_counts[zone.name] += 1
            self._count_crossing(track.track_id, position)

        return AnalyticsSnapshot(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            occupancy=len(confirmed_tracks),
            entries=self.entries,
            exits=self.exits,
            density_grid=density,
            heatmap_grid=[row.copy() for row in self._heatmap],
            zone_counts=zone_counts,
            floor_positions=positions,
            movement_vectors=movement_vectors,
            trajectories={
                track_id: list(trajectory)
                for track_id, trajectory in self.trajectories.items()
            },
        )

    def _accumulate_density(self, density: list[list[int]], point: Point) -> None:
        x, y = point
        if not (0 <= x <= self.floor_width_m and 0 <= y <= self.floor_height_m):
            return
        col = min(self.grid_cols - 1, int(x / self.floor_width_m * self.grid_cols))
        row = min(self.grid_rows - 1, int(y / self.floor_height_m * self.grid_rows))
        density[row][col] += 1

    def _count_crossing(self, track_id: int, point: Point) -> None:
        if self.counting_line is None:
            return
        side = _side_of_line(point, self.counting_line)
        if abs(side) < 1e-9:
            return
        previous = self._line_sides.get(track_id)
        previous_position = self._line_positions.get(track_id)
        self._line_sides[track_id] = side
        self._line_positions[track_id] = point
        if (
            previous is None
            or previous_position is None
            or previous * side > 0
            or not _segments_cross(
                previous_position,
                point,
                self.counting_line.start,
                self.counting_line.end,
            )
        ):
            return
        negative_to_positive = previous < 0 < side
        is_entry = (
            negative_to_positive
            if self.counting_line.entry_direction == "negative_to_positive"
            else not negative_to_positive
        )
        if is_entry:
            self.entries += 1
        else:
            self.exits += 1
