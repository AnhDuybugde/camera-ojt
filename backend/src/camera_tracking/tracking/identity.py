from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import hypot
from typing import Protocol

import numpy as np

from camera_tracking.domain import BoundingBox, Track
from camera_tracking.tracking.iou import bbox_iou


class EmbeddingExtractor(Protocol):
    def extract(self, crop_bgr: np.ndarray | None) -> np.ndarray | None: ...


@dataclass(slots=True)
class _Identity:
    stable_id: int
    bbox: BoundingBox
    embedding: np.ndarray | None
    first_seen_frame: int
    last_seen_frame: int
    hits: int = 1


class PersistentIdentityTracker:
    """Map short-lived tracker IDs to stable IDs for a fixed indoor camera."""

    def __init__(
        self,
        embedding_extractor: EmbeddingExtractor,
        *,
        max_missing_frames: int = 120,
        max_center_distance_ratio: float = 0.30,
        match_threshold: float = 0.35,
        embedding_momentum: float = 0.85,
    ) -> None:
        self.embedding_extractor = embedding_extractor
        self.max_missing_frames = max(1, max_missing_frames)
        self.max_center_distance_ratio = max(0.01, max_center_distance_ratio)
        self.match_threshold = match_threshold
        self.embedding_momentum = min(0.99, max(0.0, embedding_momentum))
        self._frame_index = 0
        self._next_stable_id = 1
        self._available_ids: list[int] = []
        self._identities: dict[int, _Identity] = {}
        self._raw_to_stable: dict[int, int] = {}

    def update(self, frame: np.ndarray, tracks: list[Track]) -> list[Track]:
        self._frame_index += 1
        self._expire_old_identities()

        embeddings = {
            track.track_id: self.embedding_extractor.extract(_crop(frame, track.bbox))
            for track in tracks
        }
        assignments: dict[int, int] = {}
        assigned_stable_ids: set[int] = set()

        # Preserve a mapping while the underlying tracker ID remains alive.
        for track in tracks:
            stable_id = self._raw_to_stable.get(track.track_id)
            if stable_id in self._identities and stable_id not in assigned_stable_ids:
                assignments[track.track_id] = stable_id
                assigned_stable_ids.add(stable_id)

        candidates: list[tuple[float, int, int]] = []
        for track in tracks:
            if track.track_id in assignments:
                continue
            embedding = embeddings[track.track_id]
            for stable_id, identity in self._identities.items():
                if stable_id in assigned_stable_ids:
                    continue
                score = self._match_score(
                    frame.shape,
                    track.bbox,
                    embedding,
                    identity,
                )
                if score >= self.match_threshold:
                    candidates.append((score, track.track_id, stable_id))

        for _, raw_id, stable_id in sorted(candidates, reverse=True):
            if raw_id in assignments or stable_id in assigned_stable_ids:
                continue
            assignments[raw_id] = stable_id
            assigned_stable_ids.add(stable_id)

        for track in tracks:
            if track.track_id in assignments:
                continue
            stable_id = self._allocate_id()
            self._identities[stable_id] = _Identity(
                stable_id=stable_id,
                bbox=track.bbox,
                embedding=embeddings[track.track_id],
                first_seen_frame=self._frame_index,
                last_seen_frame=self._frame_index,
            )
            assignments[track.track_id] = stable_id
            assigned_stable_ids.add(stable_id)

        output: list[Track] = []
        for track in tracks:
            stable_id = assignments[track.track_id]
            identity = self._identities[stable_id]
            if identity.last_seen_frame != self._frame_index:
                identity.hits += 1
            identity.bbox = track.bbox
            identity.last_seen_frame = self._frame_index
            identity.embedding = _blend_embeddings(
                identity.embedding,
                embeddings[track.track_id],
                self.embedding_momentum,
            )
            self._raw_to_stable[track.track_id] = stable_id
            output.append(
                Track(
                    track_id=stable_id,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    age=self._frame_index - identity.first_seen_frame + 1,
                    hits=identity.hits,
                    confirmed=track.confirmed,
                )
            )
        return output

    def _match_score(
        self,
        frame_shape: tuple[int, ...],
        bbox: BoundingBox,
        embedding: np.ndarray | None,
        identity: _Identity,
    ) -> float:
        frame_height, frame_width = frame_shape[:2]
        left_center = _center(bbox)
        right_center = _center(identity.bbox)
        distance = hypot(
            (left_center[0] - right_center[0]) / max(1, frame_width),
            (left_center[1] - right_center[1]) / max(1, frame_height),
        )
        if distance > self.max_center_distance_ratio:
            return 0.0

        position_score = 1.0 - distance / self.max_center_distance_ratio
        appearance_score = _cosine_similarity(embedding, identity.embedding)
        overlap_score = bbox_iou(bbox, identity.bbox)
        missing_frames = self._frame_index - identity.last_seen_frame
        recency_score = 1.0 - min(1.0, missing_frames / self.max_missing_frames)
        return (
            0.45 * position_score
            + 0.30 * appearance_score
            + 0.15 * overlap_score
            + 0.10 * recency_score
        )

    def _expire_old_identities(self) -> None:
        expired = {
            stable_id
            for stable_id, identity in self._identities.items()
            if self._frame_index - identity.last_seen_frame > self.max_missing_frames
        }
        for stable_id in expired:
            del self._identities[stable_id]
            heapq.heappush(self._available_ids, stable_id)
        if expired:
            self._raw_to_stable = {
                raw_id: stable_id
                for raw_id, stable_id in self._raw_to_stable.items()
                if stable_id not in expired
            }

    def _allocate_id(self) -> int:
        if self._available_ids:
            return heapq.heappop(self._available_ids)
        stable_id = self._next_stable_id
        self._next_stable_id += 1
        return stable_id


def _center(bbox: BoundingBox) -> tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def _crop(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1 = max(0, min(width, round(bbox.x1)))
    y1 = max(0, min(height, round(bbox.y1)))
    x2 = max(0, min(width, round(bbox.x2)))
    y2 = max(0, min(height, round(bbox.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None or left.shape != right.shape:
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return max(0.0, float(np.dot(left.ravel(), right.ravel()) / denominator))


def _blend_embeddings(
    previous: np.ndarray | None,
    current: np.ndarray | None,
    momentum: float,
) -> np.ndarray | None:
    if current is None:
        return previous
    if previous is None or previous.shape != current.shape:
        return current
    blended = momentum * previous + (1.0 - momentum) * current
    norm = float(np.linalg.norm(blended))
    return blended / norm if norm > 1e-12 else current
