from __future__ import annotations

import unicodedata
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
    show_track_label: bool = True,
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
        thickness = max(2, round(min(frame_width, frame_height) / 360))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (18, 18, 18), thickness + 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, thickness)

        if not show_track_label:
            continue

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
) -> np.ndarray:
    """Ve label hien thi: `G{gid} [| Ten] [| Label EN]`.

    - Chi hien Global ID (track.track_id da la global id tu manager).
    - Ten chi hien khi da tick diem danh (caller truyen vao).
    - Label ASCII (Working/Away/Out of office/Returning/Unknown) vi
      font Hershey cua OpenCV khong ve duoc tieng Viet co dau.
    - Khong hien ByteTrack raw ID (current ID) bao gio.
    """
    status = status or {}
    names = names or {}
    employee_ids = employee_ids or {}
    for track in tracks:
        gid = track.track_id
        parts = [f"G{gid}"]
        name = names.get(gid)
        label = getattr(status.get(gid), "label", None) if status.get(gid) else None
        if isinstance(status.get(gid), str):
            label = status.get(gid)
        if name:
            employee_id = employee_ids.get(gid)
            if employee_id:
                parts.append(f"ID {employee_id}")
            parts.append(_ascii_overlay(str(name)))
        if label:
            # A known name with Unknown means workstate is pending; it does
            # not mean face recognition failed.
            display_label = "State pending" if name and label == "Unknown" else str(label)
            parts.append(display_label)
        text = " | ".join(parts)
        text_color = status_color(label) or (0, 255, 255)
        x1, y1 = max(0, int(track.bbox.x1)), max(0, int(track.bbox.y1))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.58
        thickness = 2
        max_width = max(80, frame.shape[1] - 16)
        text = _fit_label(text, max_width - 16, font, font_scale, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        label_width = text_width + 16
        label_height = text_height + baseline + 10
        label_left = min(x1, max(0, frame.shape[1] - label_width - 1))
        label_top = y1 - label_height if y1 >= label_height else min(
            frame.shape[0] - label_height, y1 + 4
        )
        label_top = max(0, label_top)
        label_right = min(frame.shape[1] - 1, label_left + label_width)
        label_bottom = min(frame.shape[0] - 1, label_top + label_height)
        cv2.rectangle(
            frame,
            (label_left, label_top),
            (label_right, label_bottom),
            (15, 15, 15),
            -1,
        )
        cv2.rectangle(
            frame,
            (label_left + 2, label_top + 2),
            (label_right - 2, label_bottom - 2),
            text_color,
            -1,
        )
        foreground = (18, 18, 18) if _luminance(text_color) > 135 else (255, 255, 255)
        cv2.putText(
            frame,
            text,
            (label_left + 8, label_top + text_height + 5),
            font,
            font_scale,
            foreground,
            thickness,
            cv2.LINE_AA,
        )
    return frame


def draw_face_mark(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    score: float,
    known: bool,
) -> np.ndarray:
    """Draw one high-contrast face box without competing with person labels."""
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, min(frame_width - 1, round(box[0])))
    y1 = max(0, min(frame_height - 1, round(box[1])))
    x2 = max(0, min(frame_width - 1, round(box[2])))
    y2 = max(0, min(frame_height - 1, round(box[3])))
    color = (40, 180, 40) if known else (60, 60, 220)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (15, 15, 15), 5)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    label = f"FACE {'MATCH' if known else 'UNKNOWN'}  {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    label_width = text_width + 12
    label_height = text_height + baseline + 8
    label_left = min(x1, max(0, frame_width - label_width - 1))
    label_top = y1 - label_height if y1 >= label_height else y1
    label_bottom = min(frame_height - 1, label_top + label_height)
    cv2.rectangle(
        frame,
        (label_left, label_top),
        (label_left + label_width, label_bottom),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (label_left + 6, label_top + text_height + 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def _ascii_overlay(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()


def _fit_label(
    text: str,
    max_width: int,
    font: int,
    font_scale: float,
    thickness: int,
) -> str:
    if cv2.getTextSize(text, font, font_scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    candidate = text
    while candidate and cv2.getTextSize(
        candidate + suffix, font, font_scale, thickness
    )[0][0] > max_width:
        candidate = candidate[:-1]
    return candidate.rstrip() + suffix


def _luminance(color: tuple[int, int, int]) -> float:
    blue, green, red = color
    return 0.114 * blue + 0.587 * green + 0.299 * red


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
