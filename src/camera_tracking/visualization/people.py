from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from camera_tracking.domain import Track

# Box + label color by room status (BGR). Same semantics as dashboard badges.
STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "Working": (40, 180, 40),       # green
    "Near seat": (0, 200, 255),     # yellow
    "Away": (0, 200, 255),          # yellow
    "Returning": (255, 150, 0),     # blue
    "Out of office": (60, 60, 220),  # red
    "Unknown": (60, 60, 220),       # red
}
_DEFAULT_TRACK_COLOR = (200, 200, 200)


def status_color(label: str | None) -> tuple[int, int, int] | None:
    """BGR color for a room-status label, None when unknown."""
    if not label:
        return None
    return STATUS_COLORS.get(str(label).strip())


def draw_person_tracks(
    frame: np.ndarray,
    tracks: Iterable[Track],
    *,
    color: tuple[int, int, int] | None = None,
    title: str = "People",
    display_count: int | None = None,
    status: dict[int, object] | None = None,
) -> np.ndarray:
    """Draw person boxes, stable IDs, confidence, and a visible person count.

    Box color follows the room status when provided (green/yellow/red),
    otherwise falls back to the per-ID palette.
    """
    visible_tracks = list(tracks)
    frame_height, frame_width = frame.shape[:2]
    for track in visible_tracks:
        box = track.bbox
        track_color = color or _status_track_color(track.track_id, status) \
            or _track_color(track.track_id)
        x1, y1, x2, y2 = map(round, (box.x1, box.y1, box.x2, box.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, 2)

        # track_id is the shared Global ID after identity association.
        label = f"G{track.track_id}  {track.confidence:.2f}"
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


def draw_global_labels(
    frame: np.ndarray,
    tracks: Iterable[Track],
    status: dict[int, object] | None = None,
    names: dict[int, str] | None = None,
    employee_ids: dict[int, str] | None = None,
    face_candidates: dict[int, list[tuple[str, float]]] | None = None,
) -> np.ndarray:
    """Ve label hien thi: `G{gid} [| Ten] [| Label EN]`.

    - Chi hien Global ID (track.track_id da la global id tu manager).
    - Ten chi hien khi da tick diem danh (caller truyen vao).
    - Khong co mat (chua face-match) -> hien "(Unknown)", khong doan.
    - Label ASCII (Working/Away/Out of office/Returning/Unknown) vi
      font Hershey cua OpenCV khong ve duoc tieng Viet co dau.
    - Khong hien ByteTrack raw ID (current ID) bao gio.
    """
    status = status or {}
    names = names or {}
    employee_ids = employee_ids or {}
    face_candidates = face_candidates or {}
    for track in tracks:
        gid = track.track_id
        parts = [f"G{gid}"]
        name = names.get(gid)
        label = getattr(status.get(gid), "label", None) if status.get(gid) else None
        if isinstance(status.get(gid), str):
            label = status.get(gid)
        if name:
            employee_id = employee_ids.get(gid)
            # OpenCV Hershey fonts are ASCII-only; Unicode separators become
            # question marks in the live overlay.
            identity = f"ID {employee_id} - {name}" if employee_id else str(name)
            parts.append(f"({identity})")
        else:
            parts.append("(Unknown)")
        if label:
            parts.append(str(label))
        text = " | ".join(parts)
        text_color = status_color(label) or (0, 255, 255)
        x1, y1 = max(0, int(track.bbox.x1)), max(0, int(track.bbox.y1))
        cv2.putText(frame, text, (x1, max(20, y1 - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        ranked = face_candidates.get(gid, [])[:2]
        if ranked:
            ranking = " | ".join(
                f"{index}. {candidate}: {max(0.0, score) * 100:.0f}%"
                for index, (candidate, score) in enumerate(ranked, start=1)
            )
            cv2.putText(frame, ranking, (x1, max(40, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 80), 1,
                        cv2.LINE_AA)
    return frame


def _status_track_color(
    track_id: int, status: dict[int, object] | None
) -> tuple[int, int, int] | None:
    if not status:
        return None
    entry = status.get(track_id)
    label = getattr(entry, "label", None) if entry else None
    if isinstance(entry, str):
        label = entry
    return status_color(label)


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
