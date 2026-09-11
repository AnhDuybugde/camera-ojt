from __future__ import annotations

from dataclasses import dataclass

from camera_tracking.domain import BoundingBox, Detection, Track


def bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    union = left.area + right.area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox: BoundingBox
    confidence: float
    age: int = 1
    hits: int = 1
    lost_frames: int = 0


class IoUTracker:
    """Small dependency-free tracker suited to a fixed indoor camera."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 15,
        min_hits: int = 2,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.min_hits = min_hits
        self._next_id = 1
        self._states: dict[int, _TrackState] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        for state in self._states.values():
            state.age += 1
            state.lost_frames += 1

        candidates = sorted(
            (
                (bbox_iou(state.bbox, detection.bbox), track_id, detection_index)
                for track_id, state in self._states.items()
                for detection_index, detection in enumerate(detections)
            ),
            reverse=True,
        )
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        updated_ids: list[int] = []

        for score, track_id, detection_index in candidates:
            if score < self.iou_threshold:
                break
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            detection = detections[detection_index]
            state = self._states[track_id]
            state.bbox = detection.bbox
            state.confidence = detection.confidence
            state.hits += 1
            state.lost_frames = 0
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            updated_ids.append(track_id)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._states[track_id] = _TrackState(
                track_id=track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
            )
            updated_ids.append(track_id)

        expired = [
            track_id
            for track_id, state in self._states.items()
            if state.lost_frames > self.max_lost_frames
        ]
        for track_id in expired:
            del self._states[track_id]

        return [self._as_track(self._states[track_id]) for track_id in updated_ids]

    def _as_track(self, state: _TrackState) -> Track:
        return Track(
            track_id=state.track_id,
            bbox=state.bbox,
            confidence=state.confidence,
            age=state.age,
            hits=state.hits,
            confirmed=state.hits >= self.min_hits,
        )

