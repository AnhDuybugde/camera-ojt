from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from camera_tracking.domain import Track


def draw_person_tracks(
    frame: np.ndarray,
    tracks: Iterable[Track],
    *,
    color: tuple[int, int, int] | None = None,
    title: str = "People",
    display_count: int | None = None,
) -> np.ndarray:
    """Draw person boxes, stable IDs, confidence, and a visible person count."""
    visible_tracks = list(tracks)
    frame_height, frame_width = frame.shape[:2]
    for track in visible_tracks:
        box = track.bbox
        track_color = color or _track_color(track.track_id)
        x1, y1, x2, y2 = map(round, (box.x1, box.y1, box.x2, box.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, 2)

        label = f"ID {track.track_id}  {track.confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )
        label_top = max(0, y1 - label_height - baseline - 6)
        label_left = min(max(0, x1), max(0, frame_width - label_width - 8))
        cv2.rectangle(
            frame,
            (label_left, label_top),
            (label_left + label_width + 8, label_top + label_height + baseline + 6),
            track_color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (label_left + 4, label_top + label_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    count = len(visible_tracks) if display_count is None else display_count
    count_label = f"{title}: {count}"
    (count_width, _), _ = cv2.getTextSize(
        count_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    count_left = max(0, frame_width - count_width - 30)
    count_right = min(frame_width - 1, count_left + count_width + 20)
    count_bottom = min(frame_height - 1, 48)
    cv2.rectangle(frame, (count_left, 10), (count_right, count_bottom), (20, 20, 20), -1)
    cv2.putText(
        frame,
        count_label,
        (count_left + 10, min(frame_height - 5, 37)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _track_color(track_id: int) -> tuple[int, int, int]:
    palette = (
        (0, 200, 255),
        (80, 210, 90),
        (255, 170, 50),
        (210, 100, 255),
        (255, 220, 70),
        (80, 180, 255),
        (220, 130, 80),
        (120, 220, 190),
    )
    return palette[(max(1, track_id) - 1) % len(palette)]
