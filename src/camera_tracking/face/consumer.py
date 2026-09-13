from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from camera_tracking.domain import BoundingBox, Track, TrackEvent
from camera_tracking.face.embeddings import FaceDetection, FaceEmbedder, face_sharpness
from camera_tracking.face.matcher import FaceMatcher, MatchResult


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """Quality-gated face result emitted for one global track."""

    track: Track
    crop_bgr: np.ndarray
    detection: FaceDetection
    match: MatchResult
    sharpness: float
    quality: float


@dataclass(slots=True)
class _TrackGate:
    last_attempt_s: float = -float("inf")
    best_quality: float = 0.0
    employee_id: str | None = None


class FaceTrackConsumer:
    """Consume tracked person events and emit only useful face observations."""

    def __init__(
        self,
        embedder: FaceEmbedder,
        matcher: FaceMatcher,
        *,
        min_person_area_px: float = 8000.0,
        min_face_px: int = 60,
        min_blur_variance: float = 60.0,
        known_cooldown_s: float = 30.0,
        unknown_cooldown_s: float = 1.0,
    ) -> None:
        self.embedder = embedder
        self.matcher = matcher
        self.min_person_area_px = max(0.0, min_person_area_px)
        self.min_face_px = max(1, min_face_px)
        self.min_blur_variance = max(0.0, min_blur_variance)
        self.known_cooldown_s = max(0.0, known_cooldown_s)
        self.unknown_cooldown_s = max(0.0, unknown_cooldown_s)
        self._gates: dict[int, _TrackGate] = {}

    def consume(self, event: TrackEvent, now_s: float) -> list[FaceObservation]:
        if event.channel != "B":
            return []
        observations: list[FaceObservation] = []
        for track in event.tracks:
            if track.bbox.area < self.min_person_area_px:
                continue
            gate = self._gates.setdefault(_global_id(track), _TrackGate())
            cooldown = (
                self.known_cooldown_s if gate.employee_id else self.unknown_cooldown_s
            )
            if now_s - gate.last_attempt_s < cooldown:
                continue
            crop = _crop(event.frame.image, track.bbox)
            if crop is None:
                continue
            gate.last_attempt_s = now_s
            try:
                detections = self.embedder.detect_embed(crop)
            except RuntimeError:
                continue
            if not detections:
                continue
            detection = detections[0]
            face_width = detection.bbox[2] - detection.bbox[0]
            face_height = detection.bbox[3] - detection.bbox[1]
            if min(face_width, face_height) < self.min_face_px:
                continue
            face_image = _crop_box(crop, detection.bbox)
            sharpness = face_sharpness(face_image)
            if sharpness < self.min_blur_variance:
                continue
            quality = _quality(
                face_width,
                face_height,
                sharpness,
                self.min_face_px,
                self.min_blur_variance,
            )
            if quality < gate.best_quality and gate.employee_id:
                continue
            gate.best_quality = max(gate.best_quality, quality)
            match = self.matcher.match(detection.embedding)
            if match.is_known and match.person is not None:
                gate.employee_id = match.person.person_id
            observations.append(
                FaceObservation(
                    track=track,
                    crop_bgr=crop,
                    detection=detection,
                    match=match,
                    sharpness=sharpness,
                    quality=quality,
                )
            )
        return observations

    def forget_retired(self, alive_global_ids: set[int]) -> None:
        for global_id in [gid for gid in self._gates if gid not in alive_global_ids]:
            self._gates.pop(global_id, None)


def _global_id(track: Track) -> int:
    return track.global_person_id if track.global_person_id is not None else track.track_id


def _crop(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1 = max(0, min(width, round(bbox.x1)))
    y1 = max(0, min(height, round(bbox.y1)))
    x2 = max(0, min(width, round(bbox.x2)))
    y2 = max(0, min(height, round(bbox.y2)))
    return frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def _crop_box(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    return _crop(image, BoundingBox(x1, y1, x2, y2))


def _quality(
    face_width: float,
    face_height: float,
    sharpness: float,
    min_face_px: int,
    min_blur_variance: float,
) -> float:
    size_score = min(1.0, min(face_width, face_height) / max(1.0, min_face_px * 2))
    sharpness_score = min(1.0, sharpness / max(1.0, min_blur_variance * 3))
    return size_score * sharpness_score


__all__ = ["FaceObservation", "FaceTrackConsumer"]
