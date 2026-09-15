from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

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
    votes: deque[tuple[float, str | None]] = field(default_factory=deque)


class FaceTrackConsumer:
    """Consume tracked person events and emit only useful face observations."""

    def __init__(
        self,
        embedder: FaceEmbedder,
        matcher: FaceMatcher,
        *,
        min_person_area_px: float = 8000.0,
        min_face_px: int = 60,
        min_face_score: float = 0.5,
        min_blur_variance: float = 60.0,
        known_cooldown_s: float = 30.0,
        unknown_cooldown_s: float = 1.0,
        channels: tuple[str, ...] = ("A", "B"),
        consensus_hits: int = 1,
        consensus_window_s: float = 3.0,
    ) -> None:
        self.embedder = embedder
        self.matcher = matcher
        self.min_person_area_px = max(0.0, min_person_area_px)
        self.min_face_px = max(1, min_face_px)
        self.min_face_score = max(0.0, min_face_score)
        self.min_blur_variance = max(0.0, min_blur_variance)
        self.known_cooldown_s = max(0.0, known_cooldown_s)
        self.unknown_cooldown_s = max(0.0, unknown_cooldown_s)
        self.channels = tuple(channels) or ("A", "B")
        self.consensus_hits = max(1, consensus_hits)
        self.consensus_window_s = max(0.1, consensus_window_s)
        # A global identity may be visible on both cameras in the same tick.
        # Keep quality/cooldown state per camera so one channel cannot starve
        # face recognition on the other.
        self._gates: dict[tuple[str, int], _TrackGate] = {}

    def consume(self, event: TrackEvent, now_s: float) -> list[FaceObservation]:
        if event.channel not in self.channels:
            return []
        observations: list[FaceObservation] = []
        eligible: list[tuple[Track, _TrackGate]] = []
        for track in event.tracks:
            if track.bbox.area < self.min_person_area_px:
                continue
            gate_key = (event.channel, _global_id(track))
            gate = self._gates.setdefault(gate_key, _TrackGate())
            cooldown = (
                self.known_cooldown_s if gate.employee_id else self.unknown_cooldown_s
            )
            if now_s - gate.last_attempt_s < cooldown:
                continue
            gate.last_attempt_s = now_s
            eligible.append((track, gate))
        if not eligible:
            return []
        try:
            detections = self.embedder.detect_embed(event.frame.image)
        except RuntimeError:
            return []
        for track, gate, full_detection in _associate_faces(eligible, detections):
            if full_detection.score < self.min_face_score:
                continue
            crop = _crop(event.frame.image, track.bbox)
            if crop is None:
                continue
            detection = _relative_detection(full_detection, track.bbox)
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
            employee_id = (
                match.person.employee_id or match.person.person_id
                if match.is_known and match.person is not None
                else None
            )
            if gate.votes and gate.votes[-1][1] != employee_id:
                gate.votes.clear()
            gate.votes.append((now_s, employee_id))
            while gate.votes and now_s - gate.votes[0][0] > self.consensus_window_s:
                gate.votes.popleft()
            if len(gate.votes) < self.consensus_hits:
                continue
            gate.votes.clear()
            if employee_id is not None:
                gate.employee_id = employee_id
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
        retired = [
            key for key in self._gates if key[1] not in alive_global_ids
        ]
        for key in retired:
            self._gates.pop(key, None)


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


def _associate_faces(
    tracks: list[tuple[Track, _TrackGate]],
    detections: list[FaceDetection],
) -> list[tuple[Track, _TrackGate, FaceDetection]]:
    """Associate each full-frame face to at most one person's head region."""
    candidates: list[tuple[float, int, int]] = []
    for track_index, (track, _gate) in enumerate(tracks):
        box = track.bbox
        for face_index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            if not (box.x1 <= center_x <= box.x2):
                continue
            if not (box.y1 <= center_y <= box.y1 + 0.60 * box.height):
                continue
            horizontal = abs(center_x - (box.x1 + box.x2) / 2.0) / max(1.0, box.width)
            vertical = abs(center_y - (box.y1 + 0.22 * box.height)) / max(1.0, box.height)
            candidates.append((detection.score - horizontal - vertical, track_index, face_index))
    assigned_tracks: set[int] = set()
    assigned_faces: set[int] = set()
    output: list[tuple[Track, _TrackGate, FaceDetection]] = []
    for _score, track_index, face_index in sorted(candidates, reverse=True):
        if track_index in assigned_tracks or face_index in assigned_faces:
            continue
        assigned_tracks.add(track_index)
        assigned_faces.add(face_index)
        track, gate = tracks[track_index]
        output.append((track, gate, detections[face_index]))
    return output


def _relative_detection(detection: FaceDetection, person_box: BoundingBox) -> FaceDetection:
    x1, y1, x2, y2 = detection.bbox
    return FaceDetection(
        bbox=(x1 - person_box.x1, y1 - person_box.y1,
              x2 - person_box.x1, y2 - person_box.y1),
        score=detection.score,
        embedding=detection.embedding,
    )


__all__ = ["FaceObservation", "FaceTrackConsumer"]
