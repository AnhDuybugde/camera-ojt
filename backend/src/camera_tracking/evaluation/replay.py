"""Deterministic metrics for synchronized two-camera replay traces."""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from camera_tracking.domain import BoundingBox, Track

_FIELDS = (
    "frame", "camera", "global_id", "employee_id",
    "x1", "y1", "x2", "y2",
)


class IdentityTraceWriter:
    """Append production identity outputs in a replay-friendly CSV format."""

    def __init__(self, path: str | Path | None) -> None:
        self._handle = None
        self._writer = None
        if path is None:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._handle = target.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=_FIELDS)
        self._writer.writeheader()

    def write(
        self,
        frame: int,
        camera: str,
        tracks: list[Track],
        employees: dict[int, str],
    ) -> None:
        if self._writer is None:
            return
        for track in tracks:
            self._writer.writerow({
                "frame": frame,
                "camera": camera,
                "global_id": track.track_id,
                "employee_id": employees.get(track.track_id, ""),
                "x1": track.bbox.x1,
                "y1": track.bbox.y1,
                "x2": track.bbox.x2,
                "y2": track.bbox.y2,
            })
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@dataclass(frozen=True, slots=True)
class ReplayMetrics:
    matched_observations: int
    false_employee_assignments: int
    employee_uniqueness_violations: int
    id_switches: int
    promoted_global_ids: int

    def as_dict(self) -> dict[str, int]:
        return {
            "matched_observations": self.matched_observations,
            "false_employee_assignments": self.false_employee_assignments,
            "employee_uniqueness_violations": self.employee_uniqueness_violations,
            "id_switches": self.id_switches,
            "promoted_global_ids": self.promoted_global_ids,
        }


@dataclass(frozen=True, slots=True)
class _Row:
    frame: int
    camera: str
    subject_id: str
    employee_id: str
    bbox: BoundingBox


def evaluate_replay(
    truth_path: str | Path,
    prediction_path: str | Path,
    *,
    min_iou: float = 0.50,
) -> ReplayMetrics:
    """Compare trace CSV against truth CSV using per-frame box association.

    Truth columns are frame,camera,truth_id,employee_id,x1,y1,x2,y2.
    Prediction columns are emitted by :class:`IdentityTraceWriter`.
    """
    truth = _read_rows(truth_path, "truth_id")
    predictions = _read_rows(prediction_path, "global_id")
    by_key_truth = _group(truth)
    by_key_predictions = _group(predictions)
    previous_gid: dict[str, str] = {}
    employee_owners_by_frame: dict[int, dict[str, str]] = defaultdict(dict)
    seen_gids: set[str] = set()
    matched = false_employee = switches = uniqueness = 0
    for key, expected in sorted(by_key_truth.items()):
        observed = by_key_predictions.get(key, [])
        pairs = _greedy_pairs(expected, observed, min_iou)
        for actual, predicted in pairs:
            matched += 1
            seen_gids.add(predicted.subject_id)
            if predicted.employee_id and predicted.employee_id != actual.employee_id:
                false_employee += 1
            previous = previous_gid.get(actual.subject_id)
            if previous is not None and previous != predicted.subject_id:
                switches += 1
            previous_gid[actual.subject_id] = predicted.subject_id
            if predicted.employee_id:
                employee_owners = employee_owners_by_frame[actual.frame]
                owner = employee_owners.setdefault(
                    predicted.employee_id, predicted.subject_id
                )
                if owner != predicted.subject_id:
                    uniqueness += 1
    return ReplayMetrics(matched, false_employee, uniqueness, switches, len(seen_gids))


def _read_rows(path: str | Path, id_field: str) -> list[_Row]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        _Row(
            frame=int(row["frame"]),
            camera=row["camera"],
            subject_id=row[id_field],
            employee_id=row.get("employee_id", ""),
            bbox=BoundingBox(*(float(row[field]) for field in ("x1", "y1", "x2", "y2"))),
        )
        for row in rows
    ]


def _group(rows: list[_Row]) -> dict[tuple[int, str], list[_Row]]:
    grouped: dict[tuple[int, str], list[_Row]] = defaultdict(list)
    for row in rows:
        grouped[(row.frame, row.camera)].append(row)
    return grouped


def _greedy_pairs(
    truth: list[_Row], predictions: list[_Row], min_iou: float
) -> list[tuple[_Row, _Row]]:
    candidates = sorted(
        ((_iou(left.bbox, right.bbox), left_index, right_index)
         for left_index, left in enumerate(truth)
         for right_index, right in enumerate(predictions)),
        reverse=True,
    )
    used_truth: set[int] = set()
    used_predictions: set[int] = set()
    pairs: list[tuple[_Row, _Row]] = []
    for score, left_index, right_index in candidates:
        if score < min_iou:
            break
        if left_index in used_truth or right_index in used_predictions:
            continue
        used_truth.add(left_index)
        used_predictions.add(right_index)
        pairs.append((truth[left_index], predictions[right_index]))
    return pairs


def _iou(left: BoundingBox, right: BoundingBox) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = width * height
    union = left.area + right.area - intersection
    return intersection / union if union > 0 else 0.0


__all__ = ["IdentityTraceWriter", "ReplayMetrics", "evaluate_replay"]
