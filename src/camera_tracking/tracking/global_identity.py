"""Global Identity Manager: stable Global Person IDs across tracklets and channels.

ByteTrack (per channel) keeps doing short-term tracking: motion prediction,
IoU matching, data association and track buffering on brief detection gaps.
This module sits on top and answers only one question: "which person is this?".

Design points (see task spec):
- Each identity keeps an appearance *gallery* (several good Re-ID embeddings)
  instead of a single feature vector.
- Matching combines appearance similarity, observation history, time since
  last seen, appear/disappear positions, current channel and spatio-temporal
  constraints into a single score.
- Movement priors between channel 1 and channel 2 are *soft evidence only*:
  a small configurable bonus/penalty, never a mandatory route.
- Multi-track / multi-identity association is solved globally with the
  Hungarian algorithm on a cost matrix (not per-detection argmax), after a
  gating step removes implausible pairs.
- Identities are never deleted on detection loss. They move through
  ACTIVE -> TEMP_LOST -> LONG_LOST -> UNRESOLVED and can still reconnect to
  their old Global ID when the person reappears (even on another channel).
- Business logic ("is the person working / away / possibly out / returning")
  lives in a separate layer and must never write into this manager.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import hypot
from typing import Protocol

import numpy as np

from camera_tracking.domain import BoundingBox, Track

try:  # Prefer the optimal assignment solver; fall back to greedy otherwise.
    from scipy.optimize import linear_sum_assignment

    _HUNGARIAN_AVAILABLE = True
except ImportError:  # pragma: no cover - scipy ships with ultralytics.
    linear_sum_assignment = None  # type: ignore[assignment]
    _HUNGARIAN_AVAILABLE = False

#: Cost used for gated (implausible) observation <-> identity pairs.
GATED_COST = 1e6

#: Neutral spatial score for cross-channel pairs (positions are not comparable
#: across different cameras, so appearance + time decide).
CROSS_CHANNEL_SPATIAL_SCORE = 0.5


class EmbeddingExtractor(Protocol):
    def extract(self, crop_bgr: np.ndarray | None) -> np.ndarray | None: ...


class IdentityState(str, Enum):
    """Lifecycle of a global identity. Never deleted abruptly on detection loss."""

    ACTIVE = "ACTIVE"  # currently observed in at least one channel
    TEMP_LOST = "TEMP_LOST"  # briefly missing (occlusion / detector flicker)
    LONG_LOST = "LONG_LOST"  # gone longer, but still a prime reconnect candidate
    UNRESOLVED = "UNRESOLVED"  # missing for a very long time; gallery retained


@dataclass
class GlobalIdentityConfig:
    """Tunable weights and windows for the Global Identity Manager."""

    gallery_size: int = 8
    # Score weights (appearance is primary; channel prior stays a soft nudge).
    appearance_weight: float = 0.55
    spatial_weight: float = 0.20
    time_weight: float = 0.15
    channel_weight: float = 0.10
    # Minimum total score to accept an observation <-> identity match.
    match_threshold: float = 0.40
    # Gating: same-channel pairs farther apart (normalized) are implausible.
    max_center_distance_ratio: float = 0.35
    # Gating: pairs less similar than this in appearance are implausible
    # (only when both sides actually have an embedding).
    min_appearance_similarity: float = 0.30
    # Lifecycle windows (seconds when timestamps are given, else steps).
    temp_lost_s: float = 5.0
    long_lost_s: float = 60.0
    unresolved_keep_s: float = 300.0
    temp_lost_steps: int = 60
    long_lost_steps: int = 600
    unresolved_keep_steps: int = 3600
    # Only embeddings from confident, reasonably sized crops join the gallery.
    min_gallery_confidence: float = 0.25
    min_gallery_crop_pixels: int = 8
    # Soft channel-transition evidence: (from_channel, to_channel) -> bonus.
    # Small by design; appearance + spatio-temporal terms always dominate.
    channel_transition_bonus: dict[tuple[str, str], float] = field(
        default_factory=lambda: {
            ("A", "A"): 0.02,
            ("B", "B"): 0.02,
            ("A", "B"): 0.03,
            ("B", "A"): 0.03,
        }
    )


@dataclass
class IdentityRecord:
    """All global state kept for one person (one Global ID)."""

    global_id: int
    gallery: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=8))
    bbox: BoundingBox | None = None
    center: tuple[float, float] = (0.0, 0.0)
    channel: str = ""
    state: IdentityState = IdentityState.ACTIVE
    first_seen_step: int = 0
    last_seen_step: int = 0
    last_seen_s: float | None = None
    total_hits: int = 0
    disappear_center: tuple[float, float] | None = None
    disappear_channel: str = ""
    tracklet_keys: set[str] = field(default_factory=set)
    last_match_score: float = 0.0


class GlobalIdentityManager:
    """Map per-channel ByteTrack tracklets to stable cross-channel Global IDs.

    One shared instance serves all channels. Feed it with::

        manager.update(channel="A", frame=frame_a, tracks=tracks_a, now_s=t)
        manager.update(channel="B", frame=frame_b, tracks=tracks_b, now_s=t)

    Returned tracks carry ``track_id == global_id`` and can be drawn directly.
    """

    def __init__(
        self,
        embedding_extractor: EmbeddingExtractor,
        config: GlobalIdentityConfig | None = None,
    ) -> None:
        self.embedding_extractor = embedding_extractor
        self.config = config or GlobalIdentityConfig()
        self._step = 0
        self._next_global_id = 1
        self._identities: dict[int, IdentityRecord] = {}
        # Short-term continuity: "A:7" (channel + ByteTrack id) -> global id.
        self._tracklet_to_gid: dict[str, int] = {}

    # ------------------------------------------------------------------ API
    @property
    def identities(self) -> dict[int, IdentityRecord]:
        return self._identities

    def state_of(self, global_id: int) -> IdentityState:
        return self._identities[global_id].state

    def update(
        self,
        *,
        channel: str,
        frame: np.ndarray,
        tracks: list[Track],
        now_s: float | None = None,
    ) -> list[Track]:
        """Associate one channel's tracklets and return them with Global IDs."""
        self._step += 1
        self._retire_identities(now_s)
        self._refresh_states(now_s)

        embeddings = {
            track.track_id: self.embedding_extractor.extract(_crop(frame, track.bbox))
            for track in tracks
        }
        frame_shape = frame.shape
        claimed_gids: set[int] = set()
        assignments: dict[int, int] = {}  # raw track_id -> global id

        # 1. Fast path: an alive ByteTrack tracklet keeps its Global ID as long
        #    as the same-channel spatial gate still passes (no teleport).
        pending: list[Track] = []
        for track in tracks:
            key = _tracklet_key(channel, track.track_id)
            gid = self._tracklet_to_gid.get(key)
            record = self._identities.get(gid) if gid is not None else None
            if (
                record is not None
                and gid not in claimed_gids
                and self._spatial_ok(channel, track.bbox, frame_shape, record)
            ):
                assignments[track.track_id] = gid
                claimed_gids.add(gid)
                self._observe(record, channel, track, embeddings[track.track_id], now_s,
                              score=1.0)
            else:
                if key in self._tracklet_to_gid:
                    # Tracklet drifted too far: drop the shortcut so the global
                    # association can re-decide instead of forcing a wrong ID.
                    del self._tracklet_to_gid[key]
                pending.append(track)

        # 2. Global association for the rest: cost matrix + Hungarian.
        if pending:
            candidates = [
                record
                for record in self._identities.values()
                if record.global_id not in claimed_gids
            ]
            matches, unmatched_rows = self._associate(
                channel, frame_shape, pending, embeddings, candidates, now_s
            )
            for row, record, score in matches:
                track = pending[row]
                assignments[track.track_id] = record.global_id
                claimed_gids.add(record.global_id)
                self._observe(record, channel, track, embeddings[track.track_id],
                              now_s, score=score)
            for row in unmatched_rows:
                track = pending[row]
                record = self._create_identity(
                    channel, track, embeddings[track.track_id], now_s
                )
                assignments[track.track_id] = record.global_id
                claimed_gids.add(record.global_id)

        return [
            Track(
                track_id=assignments[track.track_id],
                bbox=track.bbox,
                confidence=track.confidence,
                age=self._step - self._identities[assignments[track.track_id]].first_seen_step + 1,
                hits=self._identities[assignments[track.track_id]].total_hits,
                confirmed=track.confirmed,
            )
            for track in tracks
        ]

    # ------------------------------------------------------------- matching
    def _associate(
        self,
        channel: str,
        frame_shape: tuple[int, ...],
        pending: list[Track],
        embeddings: dict[int, np.ndarray | None],
        candidates: list[IdentityRecord],
        now_s: float | None,
    ) -> tuple[list[tuple[int, IdentityRecord, float]], list[int]]:
        """Build the cost matrix, gate, solve globally, threshold.

        Returns (accepted (row, record, score) matches, unmatched row indices).
        """
        if not candidates:
            return [], list(range(len(pending)))
        n_rows, n_cols = len(pending), len(candidates)
        costs = np.full((n_rows, n_cols), GATED_COST, dtype=np.float64)
        scores = np.zeros((n_rows, n_cols), dtype=np.float64)
        for row, track in enumerate(pending):
            for col, record in enumerate(candidates):
                score = self._pair_score(
                    channel, track.bbox, frame_shape,
                    embeddings[track.track_id], record, now_s,
                )
                if score is not None:
                    scores[row, col] = score
                    costs[row, col] = 1.0 - score

        row_ind, col_ind = _solve_assignment(costs)
        matches: list[tuple[int, IdentityRecord, float]] = []
        matched_rows: set[int] = set()
        for row, col in zip(row_ind.tolist(), col_ind.tolist()):
            if costs[row, col] >= GATED_COST / 2:
                continue  # gated: implausible pair, never forced.
            score = float(scores[row, col])
            if score < self.config.match_threshold:
                continue
            matches.append((row, candidates[col], score))
            matched_rows.add(row)
        unmatched = [row for row in range(n_rows) if row not in matched_rows]
        return matches, unmatched

    def _pair_score(
        self,
        channel: str,
        bbox: BoundingBox,
        frame_shape: tuple[int, ...],
        embedding: np.ndarray | None,
        record: IdentityRecord,
        now_s: float | None,
    ) -> float | None:
        """Total match score, or None when the pair is gated out."""
        cfg = self.config
        appearance = _gallery_similarity(embedding, record.gallery)
        if (
            embedding is not None
            and len(record.gallery) > 0
            and appearance < cfg.min_appearance_similarity
        ):
            return None  # Gating: looks like a different person.

        same_channel = channel == record.channel
        distance = _normalized_distance(bbox, record, frame_shape)
        if same_channel and distance is not None and distance > cfg.max_center_distance_ratio:
            return None  # Gating: same camera, implausible jump.
        if same_channel and distance is not None:
            spatial = 1.0 - distance / cfg.max_center_distance_ratio
        else:
            spatial = CROSS_CHANNEL_SPATIAL_SCORE

        time_score = 1.0 - min(1.0, self._missing_amount(record, now_s)
                               / self._keep_window(now_s))
        channel_bonus = cfg.channel_transition_bonus.get(
            (record.channel, channel), 0.0
        )
        return (
            cfg.appearance_weight * appearance
            + cfg.spatial_weight * spatial
            + cfg.time_weight * time_score
            + cfg.channel_weight * (0.5 + channel_bonus)
        )

    # ------------------------------------------------------------ lifecycle
    def _observe(
        self,
        record: IdentityRecord,
        channel: str,
        track: Track,
        embedding: np.ndarray | None,
        now_s: float | None,
        *,
        score: float,
    ) -> None:
        if (
            embedding is not None
            and track.confidence >= self.config.min_gallery_confidence
            and max(1, int(track.bbox.width)) >= self.config.min_gallery_crop_pixels
            and max(1, int(track.bbox.height)) >= self.config.min_gallery_crop_pixels
        ):
            record.gallery.append(np.asarray(embedding, dtype=np.float32).ravel())
        record.bbox = track.bbox
        record.center = _center(track.bbox)
        record.channel = channel
        record.state = IdentityState.ACTIVE
        record.last_seen_step = self._step
        record.last_seen_s = now_s if now_s is not None else record.last_seen_s
        record.total_hits += 1
        record.disappear_center = None
        record.tracklet_keys.add(_tracklet_key(channel, track.track_id))
        record.last_match_score = score
        self._tracklet_to_gid[_tracklet_key(channel, track.track_id)] = record.global_id

    def _create_identity(
        self,
        channel: str,
        track: Track,
        embedding: np.ndarray | None,
        now_s: float | None,
    ) -> IdentityRecord:
        gid = self._next_global_id
        self._next_global_id += 1
        gallery: deque[np.ndarray] = deque(maxlen=self.config.gallery_size)
        if embedding is not None:
            gallery.append(np.asarray(embedding, dtype=np.float32).ravel())
        record = IdentityRecord(
            global_id=gid,
            gallery=gallery,
            bbox=track.bbox,
            center=_center(track.bbox),
            channel=channel,
            state=IdentityState.ACTIVE,
            first_seen_step=self._step,
            last_seen_step=self._step,
            last_seen_s=now_s,
            total_hits=1,
            tracklet_keys={_tracklet_key(channel, track.track_id)},
        )
        self._identities[gid] = record
        self._tracklet_to_gid[_tracklet_key(channel, track.track_id)] = gid
        return record

    def _refresh_states(self, now_s: float | None) -> None:
        for record in self._identities.values():
            if record.last_seen_step == self._step:
                record.state = IdentityState.ACTIVE
                continue
            if record.state == IdentityState.ACTIVE and record.disappear_center is None:
                record.disappear_center = record.center
                record.disappear_channel = record.channel
            missing = self._missing_amount(record, now_s)
            if missing <= self._temp_window(now_s):
                record.state = IdentityState.TEMP_LOST
            elif missing <= self._long_window(now_s):
                record.state = IdentityState.LONG_LOST
            else:
                record.state = IdentityState.UNRESOLVED

    def _retire_identities(self, now_s: float | None) -> None:
        """Purge only UNRESOLVED identities past the keep window (gallery freed)."""
        keep = self._keep_window(now_s)
        retired = [
            gid for gid, record in self._identities.items()
            if self._missing_amount(record, now_s) > keep
            and record.state == IdentityState.UNRESOLVED
        ]
        for gid in retired:
            del self._identities[gid]
            self._tracklet_to_gid = {
                key: value for key, value in self._tracklet_to_gid.items()
                if value != gid
            }

    # -------------------------------------------------------------- helpers
    def _spatial_ok(
        self,
        channel: str,
        bbox: BoundingBox,
        frame_shape: tuple[int, ...],
        record: IdentityRecord,
    ) -> bool:
        if channel != record.channel:
            return True  # cross-channel pairs are decided by global matching.
        distance = _normalized_distance(bbox, record, frame_shape)
        return distance is not None and distance <= self.config.max_center_distance_ratio

    def _missing_amount(self, record: IdentityRecord, now_s: float | None) -> float:
        if now_s is not None and record.last_seen_s is not None:
            return max(0.0, now_s - record.last_seen_s)
        return float(self._step - record.last_seen_step)

    def _temp_window(self, now_s: float | None) -> float:
        return self.config.temp_lost_s if now_s is not None else float(self.config.temp_lost_steps)

    def _long_window(self, now_s: float | None) -> float:
        return self.config.long_lost_s if now_s is not None else float(self.config.long_lost_steps)

    def _keep_window(self, now_s: float | None) -> float:
        return self.config.unresolved_keep_s if now_s is not None else float(self.config.unresolved_keep_steps)


def _solve_assignment(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if _HUNGARIAN_AVAILABLE:
        return linear_sum_assignment(costs)
    # Greedy fallback: repeatedly take the cheapest remaining pair.
    n_rows, n_cols = costs.shape
    remaining_rows = set(range(n_rows))
    remaining_cols = set(range(n_cols))
    row_ind: list[int] = []
    col_ind: list[int] = []
    flat = sorted(
        (float(costs[r, c]), r, c) for r in remaining_rows for c in remaining_cols
    )
    for _, row, col in flat:
        if row in remaining_rows and col in remaining_cols:
            row_ind.append(row)
            col_ind.append(col)
            remaining_rows.discard(row)
            remaining_cols.discard(col)
    return np.asarray(row_ind), np.asarray(col_ind)


def _tracklet_key(channel: str, raw_track_id: int) -> str:
    # ByteTrack ids are per-tracker instances; channel A id 1 and channel B
    # id 1 are different tracklets, hence the namespace.
    return f"{channel}:{raw_track_id}"


def _center(bbox: BoundingBox) -> tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def _normalized_distance(
    bbox: BoundingBox,
    record: IdentityRecord,
    frame_shape: tuple[int, ...],
) -> float | None:
    if record.bbox is None:
        return None
    frame_height, frame_width = frame_shape[:2]
    left = _center(bbox)
    return hypot(
        (left[0] - record.center[0]) / max(1, frame_width),
        (left[1] - record.center[1]) / max(1, frame_height),
    )


def _gallery_similarity(
    embedding: np.ndarray | None, gallery: deque[np.ndarray]
) -> float:
    """Best cosine similarity against the gallery (robust to pose changes)."""
    if embedding is None or not gallery:
        return 0.0
    query = np.asarray(embedding, dtype=np.float32).ravel()
    best = 0.0
    for stored in gallery:
        best = max(best, _cosine_similarity(query, stored))
    return best


def _cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    left = np.asarray(left, dtype=np.float32).ravel()
    right = np.asarray(right, dtype=np.float32).ravel()
    if left.shape != right.shape or left.size == 0:
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return max(0.0, float(np.dot(left, right) / denominator))


def _crop(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1 = max(0, min(width, round(bbox.x1)))
    y1 = max(0, min(height, round(bbox.y1)))
    x2 = max(0, min(width, round(bbox.x2)))
    y2 = max(0, min(height, round(bbox.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


__all__ = [
    "CROSS_CHANNEL_SPATIAL_SCORE",
    "GATED_COST",
    "EmbeddingExtractor",
    "GlobalIdentityConfig",
    "GlobalIdentityManager",
    "IdentityRecord",
    "IdentityState",
]
