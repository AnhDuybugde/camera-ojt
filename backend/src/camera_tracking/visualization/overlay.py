from __future__ import annotations

from itertools import pairwise

import cv2
import numpy as np

from camera_tracking.analytics import CountingLine, FloorProjector, Zone
from camera_tracking.domain import AnalyticsSnapshot, Point, Track


class OverlayRenderer:
    def __init__(
        self,
        projector: FloorProjector,
        floor_width_m: float,
        floor_height_m: float,
        zones: list[Zone] | None = None,
        counting_line: CountingLine | None = None,
    ) -> None:
        self.projector = projector
        self.floor_width_m = floor_width_m
        self.floor_height_m = floor_height_m
        self.zones = zones or []
        self.counting_line = counting_line

    def render(
        self, image: np.ndarray, tracks: list[Track], snapshot: AnalyticsSnapshot
    ) -> np.ndarray:
        canvas = image.copy()
        for track in tracks:
            if not track.confirmed:
                continue
            box = track.bbox
            color = (54, 211, 153)
            cv2.rectangle(
                canvas,
                (round(box.x1), round(box.y1)),
                (round(box.x2), round(box.y2)),
                color,
                2,
            )
            cv2.putText(
                canvas,
                f"ID {track.track_id} {track.confidence:.2f}",
                (round(box.x1), max(20, round(box.y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            trajectory = snapshot.trajectories.get(track.track_id, [])
            image_points = [self.projector.unproject(point) for point in trajectory]
            for start, end in pairwise(image_points):
                cv2.line(canvas, _pixel(start), _pixel(end), (0, 190, 255), 2)

        panel_width = min(310, canvas.shape[1])
        cv2.rectangle(canvas, (0, 0), (panel_width, 88), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            f"Occupancy: {snapshot.occupancy}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"Entries: {snapshot.entries}  Exits: {snapshot.exits}",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )
        self._draw_floor_map(canvas, snapshot)
        return canvas

    def _draw_floor_map(self, canvas: np.ndarray, snapshot: AnalyticsSnapshot) -> None:
        map_width = min(260, max(120, canvas.shape[1] // 4))
        map_height = max(90, round(map_width * self.floor_height_m / self.floor_width_m))
        map_height = min(map_height, max(90, canvas.shape[0] // 2))
        margin = 12
        left = max(0, canvas.shape[1] - map_width - margin)
        top = max(0, canvas.shape[0] - map_height - margin)
        right = left + map_width
        bottom = top + map_height
        cv2.rectangle(canvas, (left, top), (right, bottom), (28, 28, 28), -1)

        rows = len(snapshot.heatmap_grid)
        cols = len(snapshot.heatmap_grid[0]) if rows else 0
        maximum = max((max(row) for row in snapshot.heatmap_grid), default=0)
        for row_index, row in enumerate(snapshot.heatmap_grid):
            for col_index, count in enumerate(row):
                intensity = count / maximum if maximum else 0
                color = (35, round(80 + 120 * intensity), round(40 + 200 * intensity))
                x1 = left + round(col_index / cols * map_width)
                x2 = left + round((col_index + 1) / cols * map_width)
                display_row = rows - row_index - 1
                y1 = top + round(display_row / rows * map_height)
                y2 = top + round((display_row + 1) / rows * map_height)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)

        def floor_pixel(point: Point) -> tuple[int, int]:
            x = left + round(point[0] / self.floor_width_m * map_width)
            y = bottom - round(point[1] / self.floor_height_m * map_height)
            return (x, y)

        for zone in self.zones:
            polygon = np.asarray([floor_pixel(point) for point in zone.points], np.int32)
            if len(polygon) >= 3:
                cv2.polylines(canvas, [polygon], True, (255, 210, 80), 1, cv2.LINE_AA)
        if self.counting_line:
            cv2.line(
                canvas,
                floor_pixel(self.counting_line.start),
                floor_pixel(self.counting_line.end),
                (80, 80, 255),
                2,
            )
        for track_id, trajectory in snapshot.trajectories.items():
            points = [floor_pixel(point) for point in trajectory]
            for start, end in pairwise(points):
                cv2.line(canvas, start, end, (0, 0, 0), 2, cv2.LINE_AA)
            if track_id in snapshot.floor_positions:
                cv2.circle(canvas, points[-1], 4, (255, 255, 255), -1)
        cv2.rectangle(canvas, (left, top), (right, bottom), (230, 230, 230), 1)


def _pixel(point: tuple[float, float]) -> tuple[int, int]:
    return (round(point[0]), round(point[1]))
