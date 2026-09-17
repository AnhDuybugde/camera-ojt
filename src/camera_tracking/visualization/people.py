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
_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


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
    focus_gid: int | None = None,
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
        focused = focus_gid is None or track.track_id == focus_gid
        thickness = 3 if focus_gid is not None and focused else (2 if focused else 1)
        if not focused:
            track_color = tuple(int(value * 0.55) for value in track_color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, thickness)

        # track_id is the shared Global ID after identity association.
        if focus_gid is not None and not focused:
            entry = (status or {}).get(track.track_id)
            short_status = getattr(entry, "label", None)
            if isinstance(entry, str):
                short_status = entry
            label = f"G{track.track_id}" + (
                f"  {short_status}" if short_status else "")
        else:
            label = (f"FOCUS G{track.track_id}  {track.confidence:.2f}"
                     if focus_gid is not None
                     else f"G{track.track_id}  {track.confidence:.2f}")
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
    focus_gid: int | None = None,
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
        if focus_gid is not None and gid != focus_gid:
            # Background recognition continues, but its result must not steal
            # the primary UI. The thin track box/GID remains for diagnostics.
            continue
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


def draw_hand_landmarks(
    frame: np.ndarray,
    person_bbox,
    hands: list[list[tuple[float, float]]],
    valid: list[bool] | None = None,
    *,
    reversals: int = 0,
    required_reversals: int = 0,
) -> np.ndarray:
    """Draw MediaPipe hand skeletons mapped from a person crop to the frame.

    Green means the hand passed the open-palm activation gate; orange means
    MediaPipe saw a hand but it has not satisfied all gesture conditions.
    """
    height, width = frame.shape[:2]
    x1 = max(0, min(width, round(person_bbox.x1)))
    y1 = max(0, min(height, round(person_bbox.y1)))
    x2 = max(0, min(width, round(person_bbox.x2)))
    y2 = max(0, min(height, round(person_bbox.y2)))
    crop_w, crop_h = x2 - x1, y2 - y1
    if crop_w <= 0 or crop_h <= 0:
        return frame
    accepted = valid or []
    for hand_index, hand in enumerate(hands):
        if len(hand) < 21:
            continue
        color = ((40, 220, 40)
                 if hand_index < len(accepted) and accepted[hand_index]
                 else (0, 165, 255))
        points = [
            (int(x1 + max(0.0, min(1.0, px)) * crop_w),
             int(y1 + max(0.0, min(1.0, py)) * crop_h))
            for px, py in hand
        ]
        for start, end in _HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)
        for index, point in enumerate(points):
            radius = 4 if index in (0, 4, 8, 12, 16, 20) else 3
            cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
        fingers = _visible_extended_fingers(hand)
        state = "OPEN" if hand_index < len(accepted) and accepted[hand_index] else "HAND"
        cv2.putText(
            frame,
            f"{state} {fingers}/5 | WAVE {reversals}/{required_reversals}",
            (points[0][0], min(height - 5, points[0][1] + 22)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 2, cv2.LINE_AA,
        )
    return frame


def _visible_extended_fingers(hand: list[tuple[float, float]]) -> int:
    """Small local diagnostic equivalent of the gesture finger counter."""
    if len(hand) < 21:
        return 0
    wrist = hand[0]

    def distance(a, b) -> float:
        return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)

    count = sum(
        distance(hand[tip], wrist) > distance(hand[pip], wrist) * 1.08
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    )
    count += int(
        distance(hand[4], wrist) > distance(hand[3], wrist) * 1.03
        and abs(hand[4][0] - hand[5][0]) >= 0.04
    )
    return count


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
