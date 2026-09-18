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
    match_threshold: float = 0.70
    # Gating: same-channel pairs farther apart (normalized) are implausible.
    max_center_distance_ratio: float = 0.35
    same_camera_reconnect_distance_ratio: float = 0.08
    same_camera_reconnect_s: float = 2.0
    # Gating: pairs less similar than this in appearance are implausible
    # (only when both sides actually have an embedding).
    min_appearance_similarity: float = 0.70
    # Stricter appearance floor for reusing a named (employee-bound) identity.
    # Ngăn GID đã gắn tên bị body khác mặc đồ khác chiếm chỉ vì đứng gần.
    named_appearance_floor: float = 0.70
    # Even stricter floor while a fresh high-score face anchors the identity
    # (see set_face_anchor). Ngăn B cướp GID của A ngay sau khi A vừa được
    # face xác nhận điểm cao rồi bị che khuất.
    named_floor_locked: float = 0.80
    # Face anchor TTL: bao lâu sau 1 face score cao thì GID được "khóa".
    face_anchor_ttl_s: float = 10.0
    face_anchor_ttl_steps: int = 120
    # Occlusion suspicion: IoU giữa 2 track cùng frame vượt ngưỡng này thì
    # fast-path phải verify appearance thay vì giữ mù.
    occlusion_iou_threshold: float = 0.40
    # Area jump tương đối so với sighting cũ vượt ngưỡng này cũng nghi ngờ.
    occlusion_area_jump_ratio: float = 0.30
    # Gallery hygiene: embedding mới khác gallery hiện tại quá thì không append.
    gallery_append_min_sim: float = 0.60
    # Strong evidence for merging two simultaneously active cross-camera IDs.
    active_duplicate_similarity: float = 0.90
    tentative_min_hits: int = 1
    # Lifecycle windows (seconds when timestamps are given, else steps).
    temp_lost_s: float = 5.0
    long_lost_s: float = 60.0
    unresolved_keep_s: float = 86400.0
    temp_lost_steps: int = 60
    long_lost_steps: int = 600
    unresolved_keep_steps: int = 3600
    # Only embeddings from confident, reasonably sized crops join the gallery.
    min_gallery_confidence: float = 0.25
    min_gallery_crop_pixels: int = 8
    # Perf: stable tracklets (fast path) refresh the gallery only every N
    # manager steps instead of every frame. New/lost tracklets (pending
    # path) always extract -- matching needs fresh appearance. 1 = old
    # behavior (extract for every track every frame).
    gallery_refresh_steps: int = 1
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
class CameraSighting:
    """Latest observation of one identity on one camera."""

    bbox: BoundingBox
    center: tuple[float, float]
    last_seen_step: int
    last_seen_s: float | None


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
    employee_id: str | None = None
    sightings: dict[str, CameraSighting] = field(default_factory=dict)
    # Occlusion / face-anchor state (transient, không persist qua ngày).
    # last_good_embedding: embedding gần nhất đã verify khớp gallery.
    last_good_embedding: np.ndarray | None = None
    # True khi lần observe gần nhất bị nghi hijack/che khuất.
    occluded: bool = False
    # Face anchor: GID vừa được face điểm cao xác nhận -> "khóa" tạm thời.
    face_anchor_employee: str | None = None
    face_anchor_score: float = 0.0
    face_anchor_s: float | None = None
    face_anchor_step: int = 0


@dataclass
class _TentativeIdentity:
    channel: str
    raw_track_id: int
    bbox: BoundingBox
    embedding: np.ndarray | None
    hits: int
    last_seen_step: int
    last_seen_s: float | None


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
        self._tentative: dict[str, _TentativeIdentity] = {}

    # ------------------------------------------------------------------ API
    @property
    def identities(self) -> dict[int, IdentityRecord]:
        return self._identities

    def state_of(self, global_id: int) -> IdentityState:
        return self._identities[global_id].state

    def reset_for_new_day(self) -> None:
        """Start a fresh public GID namespace at the local day boundary."""
        self._step = 0
        self._next_global_id = 1
        self._identities.clear()
        self._tracklet_to_gid.clear()
        self._tentative.clear()

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

        # Lazy embeddings: stable tracklets on the fast path usually need
        # no fresh appearance, so extraction is cached per call and skipped
        # unless the gallery is due for a refresh. Pending (new/lost)
        # tracklets always extract -- global matching depends on it.
        embeddings: dict[int, np.ndarray | None] = {}
        refresh_gallery = (
            self._step % max(1, self.config.gallery_refresh_steps) == 0
        )
        frame_shape = frame.shape
        claimed_gids: set[int] = set()
        assignments: dict[int, int] = {}  # raw track_id -> global id

        # 1. Fast path: an alive ByteTrack tracklet keeps its Global ID as long
        #    as the same-channel spatial gate still passes (no teleport).
        #    Anti-hijack: khi nghi occlusion (2 box đè nhau / area jump) hoặc
        #    GID đang face-locked, bắt buộc verify appearance. Nếu embedding
        #    khác gallery quá thì đá xuống pending để Hungarian quyết định lại
        #    thay vì giữ mù (B cướp GID + tên của A khi che hoàn toàn).
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
                suspect = self._is_occlusion_suspect(
                    track, tracks, record, frame_shape, channel
                )
                locked = self._is_face_locked(record, now_s)
                # Named GID (đã có employee) luôn verify: số người có tên ít
                # nên tốn thêm ReID mỗi frame là chấp nhận được, đổi lại bắt
                # được cả case B thay A khớp khít box mà không gây IoU/area jump.
                need_verify = (
                    refresh_gallery or suspect or locked or record.occluded
                    or record.employee_id is not None
                )
                if not need_verify:
                    assignments[track.track_id] = gid
                    claimed_gids.add(gid)
                    self._observe(record, channel, track, None, now_s,
                                  score=1.0)
                    continue
                embedding = self._cached_embedding(frame, track, embeddings)
                if embedding is None:
                    # Không verify được mà đã có gallery history + (nghi ngờ
                    # hoặc named/locked) thì không giữ mù -> pending.
                    if record.gallery and (
                        suspect or locked or record.employee_id
                    ):
                        record.occluded = True
                        if key in self._tracklet_to_gid:
                            del self._tracklet_to_gid[key]
                        pending.append(track)
                        continue
                    assignments[track.track_id] = gid
                    claimed_gids.add(gid)
                    self._observe(record, channel, track, None, now_s,
                                  score=1.0)
                    continue
                floor = self._effective_appearance_floor(record, now_s)
                sim = (
                    _gallery_similarity(embedding, record.gallery)
                    if record.gallery else 1.0
                )
                if record.gallery and sim < floor:
                    # Appearance đổi rõ trong khi vị trí vẫn gần -> khả năng
                    # cao là người khác đã che/nhận box -> pending.
                    record.occluded = True
                    if key in self._tracklet_to_gid:
                        del self._tracklet_to_gid[key]
                    pending.append(track)
                    continue
                assignments[track.track_id] = gid
                claimed_gids.add(gid)
                self._observe(record, channel, track, embedding, now_s,
                              score=1.0)
            else:
                if key in self._tracklet_to_gid:
                    # Tracklet drifted too far: drop the shortcut so the global
                    # association can re-decide instead of forcing a wrong ID.
                    del self._tracklet_to_gid[key]
                pending.append(track)

        # 2. Global association for the rest: cost matrix + Hungarian.
        if pending:
            # Pending tracklets always need fresh appearance for matching.
            for track in pending:
                self._cached_embedding(frame, track, embeddings)
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
                record = self._promote_tentative(
                    channel, track, embeddings[track.track_id], now_s
                )
                if record is None:
                    continue
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
                local_track_id=(
                    track.local_track_id
                    if track.local_track_id is not None
                    else track.track_id
                ),
                global_person_id=assignments[track.track_id],
                employee_id=self._identities[assignments[track.track_id]].employee_id,
            )
            for track in tracks
            if track.track_id in assignments
        ]

    def update_batch(
        self,
        observations: dict[str, tuple[np.ndarray, list[Track]]],
        *,
        now_s: float | None = None,
    ) -> dict[str, list[Track]]:
        """Update a synchronized set of cameras while preserving overlap."""
        return {
            channel: self.update(
                channel=channel, frame=frame, tracks=tracks, now_s=now_s
            )
            for channel, (frame, tracks) in observations.items()
        }

    def bind_employee(self, global_id: int, employee_id: str) -> bool:
        """Attach one employee to exactly one live Global ID.

        Global IDs describe session-level association; employee IDs are the
        durable face result. Keeping the binding here makes that distinction
        available to downstream consumers without making face recognition a
        prerequisite for tracking. Any previous binding of the same employee
        is removed atomically before the new binding is installed.
        """
        record = self._identities.get(global_id)
        if record is None or not employee_id:
            return False
        if record.employee_id not in (None, employee_id):
            return False
        if any(
            other_gid != global_id and other.employee_id == employee_id
            for other_gid, other in self._identities.items()
        ):
            return False
        record.employee_id = employee_id
        return True

    def employee_id_of(self, global_id: int) -> str | None:
        record = self._identities.get(global_id)
        return record.employee_id if record is not None else None

    def unbind_employee(self, global_id: int) -> bool:
        """Remove a wrong employee binding (face conflict loser keeps no name).

        Returns True when something was actually unbound.
        """
        record = self._identities.get(global_id)
        if record is None or not record.employee_id:
            return False
        record.employee_id = None
        # Hủy face anchor đi cùng để GID thua conflict không giữ khóa.
        record.face_anchor_employee = None
        record.face_anchor_score = 0.0
        return True

    def restore_identity(
        self,
        global_id: int,
        *,
        employee_id: str | None,
        gallery: list[np.ndarray],
    ) -> None:
        """Restore a day identity as an unresolved ReID candidate."""
        normalized = [item for item in (_normalize_embedding(v) for v in gallery)
                      if item is not None]
        if employee_id and any(
            record.employee_id == employee_id for record in self._identities.values()
        ):
            employee_id = None
        self._identities[global_id] = IdentityRecord(
            global_id=global_id,
            gallery=deque(normalized, maxlen=self.config.gallery_size),
            state=IdentityState.UNRESOLVED,
            last_seen_s=0.0,
            employee_id=employee_id,
        )
        self._next_global_id = max(self._next_global_id, global_id + 1)

    def merge_identity(self, duplicate_gid: int, canonical_gid: int) -> bool:
        """Redirect a duplicate Global ID into an older canonical ID."""
        if duplicate_gid == canonical_gid:
            return False
        duplicate = self._identities.get(duplicate_gid)
        canonical = self._identities.get(canonical_gid)
        if duplicate is None or canonical is None:
            return False
        if (
            duplicate.employee_id
            and canonical.employee_id
            and duplicate.employee_id != canonical.employee_id
        ):
            return False
        if canonical.employee_id is None:
            canonical.employee_id = duplicate.employee_id
        canonical.gallery.extend(duplicate.gallery)
        canonical.tracklet_keys.update(duplicate.tracklet_keys)
        canonical.sightings.update(duplicate.sightings)
        canonical.total_hits += duplicate.total_hits
        if duplicate.last_seen_step > canonical.last_seen_step:
            canonical.bbox = duplicate.bbox
            canonical.center = duplicate.center
            canonical.channel = duplicate.channel
            canonical.last_seen_step = duplicate.last_seen_step
            canonical.last_seen_s = duplicate.last_seen_s
            canonical.last_match_score = duplicate.last_match_score
        for key, gid in list(self._tracklet_to_gid.items()):
            if gid == duplicate_gid:
                self._tracklet_to_gid[key] = canonical_gid
        del self._identities[duplicate_gid]
        return True

    def reconcile_active_duplicates(self) -> dict[int, int]:
        """Merge very-high-similarity active identities across cameras."""
        active = [
            record for record in self._identities.values()
            if record.state in (IdentityState.ACTIVE, IdentityState.TEMP_LOST)
        ]
        aliases: dict[int, int] = {}
        for index, left in enumerate(active):
            if left.global_id not in self._identities:
                continue
            for right in active[index + 1:]:
                if right.global_id not in self._identities:
                    continue
                # Same-camera duplicates are possible when detector output
                # contains a nested box or ByteTrack fragments one person.
                # Only merge them when their boxes genuinely overlap; this
                # prevents two nearby, similarly dressed people from being
                # collapsed into one Global ID.
                if (
                    left.channel == right.channel
                    and not _same_camera_duplicate(left, right)
                ):
                    continue
                if (
                    left.employee_id
                    and right.employee_id
                    and left.employee_id != right.employee_id
                ):
                    continue
                similarity = _gallery_pair_similarity(left.gallery, right.gallery)
                if similarity < self.config.active_duplicate_similarity:
                    continue
                verified = [
                    record.global_id for record in (left, right)
                    if record.employee_id
                ]
                canonical = min(verified or [left.global_id, right.global_id])
                duplicate = right.global_id if canonical == left.global_id else left.global_id
                if self.merge_identity(duplicate, canonical):
                    aliases[duplicate] = canonical
        return aliases

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
        same_sighting = record.sightings.get(channel)
        distance = _normalized_distance_to_sighting(
            bbox, same_sighting, frame_shape
        )
        locked = self._is_face_locked(record, now_s)
        if embedding is None or not record.gallery:
            # Face-locked GID không được reconnect thuần spatial khi mất
            # embedding: đúng lúc che khuất ReID hay None nhất, cho spatial
            # là cho B mượn luôn GID + tên của A.
            if locked:
                return None
            if (
                same_sighting is not None
                and distance is not None
                and distance <= cfg.same_camera_reconnect_distance_ratio
                and self._missing_amount(record, now_s) <= cfg.same_camera_reconnect_s
            ):
                return cfg.match_threshold
            return None
        floor = self._effective_appearance_floor(record, now_s)
        if (
            embedding is not None
            and len(record.gallery) > 0
            and appearance < floor
        ):
            return None  # Gating: looks like a different person.
        # thi khong duoc gán cho tracklet kenh nay -> tach GID moi.
        same_channel = same_sighting is not None
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
    def _cached_embedding(
        self,
        frame: np.ndarray,
        track: Track,
        cache: dict[int, np.ndarray | None],
    ) -> np.ndarray | None:
        """Extract once per update() call; fast path usually skips this."""
        if track.track_id not in cache:
            cache[track.track_id] = self.embedding_extractor.extract(
                _crop(frame, track.bbox)
            )
        return cache[track.track_id]

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
            normalized = _normalize_embedding(embedding)
            if normalized is not None:
                # Gallery hygiene: embedding mới quá khác gallery hiện tại
                # (B đè box A) thì không append để tránh pollution khiến lần
                # sau càng dễ match nhầm. Vẫn update vị trí để giữ continuity.
                if record.gallery:
                    sim = _gallery_similarity(embedding, record.gallery)
                    if sim >= self.config.gallery_append_min_sim:
                        record.gallery.append(normalized)
                        record.last_good_embedding = normalized
                        record.occluded = False
                    else:
                        record.occluded = True
                else:
                    record.gallery.append(normalized)
                    record.last_good_embedding = normalized
                    record.occluded = False
        elif embedding is not None:
            # Embedding có nhưng crop/conf quá yếu -> không append nhưng cũng
            # không đánh dấu occluded (chỉ là quan sát kém).
            pass
        else:
            # Không có embedding mới: giữ nguyên gallery, không đổi cờ occluded
            # ở đây (fast-path đã quyết định trước khi gọi).
            pass
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
        record.sightings[channel] = CameraSighting(
            bbox=track.bbox,
            center=record.center,
            last_seen_step=self._step,
            last_seen_s=now_s,
        )
        self._tracklet_to_gid[_tracklet_key(channel, track.track_id)] = record.global_id

    def _promote_tentative(
        self,
        channel: str,
        track: Track,
        embedding: np.ndarray | None,
        now_s: float | None,
    ) -> IdentityRecord | None:
        key = _tracklet_key(channel, track.track_id)
        candidate = self._tentative.get(key)
        if candidate is None:
            candidate = _TentativeIdentity(
                channel=channel,
                raw_track_id=track.track_id,
                bbox=track.bbox,
                embedding=embedding,
                hits=1,
                last_seen_step=self._step,
                last_seen_s=now_s,
            )
            self._tentative[key] = candidate
        else:
            candidate.bbox = track.bbox
            if embedding is not None:
                candidate.embedding = embedding
            candidate.hits += 1
            candidate.last_seen_step = self._step
            candidate.last_seen_s = now_s
        if candidate.hits < max(1, self.config.tentative_min_hits):
            return None
        self._tentative.pop(key, None)
        return self._create_identity(channel, track, candidate.embedding, now_s)

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
        last_good: np.ndarray | None = None
        if embedding is not None:
            normalized = _normalize_embedding(embedding)
            if normalized is not None:
                gallery.append(normalized)
                last_good = normalized
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
            last_good_embedding=last_good,
            tracklet_keys={_tracklet_key(channel, track.track_id)},
            sightings={
                channel: CameraSighting(
                    bbox=track.bbox,
                    center=_center(track.bbox),
                    last_seen_step=self._step,
                    last_seen_s=now_s,
                )
            },
        )
        self._identities[gid] = record
        self._tracklet_to_gid[_tracklet_key(channel, track.track_id)] = gid
        return record

    def _refresh_states(self, now_s: float | None) -> None:
        stale_tentative = [
            key for key, candidate in self._tentative.items()
            if (
                max(0.0, now_s - candidate.last_seen_s)
                if now_s is not None and candidate.last_seen_s is not None
                else float(self._step - candidate.last_seen_step)
            ) > self._temp_window(now_s)
        ]
        for key in stale_tentative:
            self._tentative.pop(key, None)
        for record in self._identities.values():
            for channel, sighting in list(record.sightings.items()):
                missing = (
                    max(0.0, now_s - sighting.last_seen_s)
                    if now_s is not None and sighting.last_seen_s is not None
                    else float(self._step - sighting.last_seen_step)
                )
                if missing > self._temp_window(now_s):
                    record.sightings.pop(channel, None)
            missing = self._missing_amount(record, now_s)
            if record.last_seen_step == self._step or missing <= 1e-9:
                record.state = IdentityState.ACTIVE
                continue
            if record.state == IdentityState.ACTIVE and record.disappear_center is None:
                record.disappear_center = record.center
                record.disappear_channel = record.channel
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
        sighting = record.sightings.get(channel)
        if sighting is None:
            return True  # cross-channel pairs are decided by global matching.
        distance = _normalized_distance_to_sighting(bbox, sighting, frame_shape)
        return distance is not None and distance <= self.config.max_center_distance_ratio

    def set_face_anchor(
        self,
        global_id: int,
        employee_id: str,
        score: float,
        now_s: float | None = None,
    ) -> bool:
        """Khóa tạm thời GID vừa được face điểm cao xác nhận.

        Trong TTL, body lạ muốn claim GID này phải vượt
        ``named_floor_locked`` (cao hơn floor thường) và không được
        reconnect thuần spatial khi mất embedding.
        """
        record = self._identities.get(global_id)
        if record is None or not employee_id:
            return False
        record.face_anchor_employee = employee_id
        record.face_anchor_score = float(score)
        record.face_anchor_s = now_s
        record.face_anchor_step = self._step
        return True

    def _is_face_locked(
        self, record: IdentityRecord, now_s: float | None
    ) -> bool:
        if not record.face_anchor_employee:
            return False
        # Chỉ khóa đúng employee đang giữ (tránh khóa nhầm sau unbind).
        if (
            record.employee_id is not None
            and record.face_anchor_employee != record.employee_id
        ):
            return False
        if now_s is not None and record.face_anchor_s is not None:
            return (now_s - record.face_anchor_s) <= max(
                0.0, self.config.face_anchor_ttl_s
            )
        return (self._step - record.face_anchor_step) <= max(
            1, self.config.face_anchor_ttl_steps
        )

    def _effective_appearance_floor(
        self, record: IdentityRecord, now_s: float | None
    ) -> float:
        cfg = self.config
        if self._is_face_locked(record, now_s):
            return max(cfg.min_appearance_similarity,
                       cfg.named_appearance_floor,
                       cfg.named_floor_locked)
        if record.employee_id:
            return max(cfg.min_appearance_similarity,
                       cfg.named_appearance_floor)
        return cfg.min_appearance_similarity

    def _is_occlusion_suspect(
        self,
        track: Track,
        tracks: list[Track],
        record: IdentityRecord,
        frame_shape: tuple[int, ...] | None = None,
        channel: str | None = None,
    ) -> bool:
        """Nghi che khuất khi box đè nhau hoặc area jump bất thường."""
        cfg = self.config
        # 1. Cùng frame có box khác IoU cao với track hiện tại (2 người đè).
        for other in tracks:
            if other.track_id == track.track_id:
                continue
            iou = _bbox_iou(track.bbox, other.bbox)
            if iou >= cfg.occlusion_iou_threshold:
                return True
        # 2. Diện tích box đổi đột ngột so với sighting cũ (A bị B che nửa).
        lookup = channel or record.channel
        sighting = record.sightings.get(lookup) if lookup else None
        if sighting is not None and sighting.bbox is not None:
            old_area = max(1.0, sighting.bbox.area)
            new_area = max(1.0, track.bbox.area)
            jump = abs(new_area - old_area) / old_area
            if jump >= cfg.occlusion_area_jump_ratio:
                return True
        return False

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


def _normalized_distance_to_sighting(
    bbox: BoundingBox,
    sighting: CameraSighting | None,
    frame_shape: tuple[int, ...],
) -> float | None:
    if sighting is None:
        return None
    frame_height, frame_width = frame_shape[:2]
    left = _center(bbox)
    return hypot(
        (left[0] - sighting.center[0]) / max(1, frame_width),
        (left[1] - sighting.center[1]) / max(1, frame_height),
    )


def _gallery_similarity(
    embedding: np.ndarray | None, gallery: deque[np.ndarray]
) -> float:
    """Best cosine similarity against the gallery (robust to pose changes)."""
    if embedding is None or not gallery:
        return 0.0
    query = _normalize_embedding(embedding)
    if query is None:
        return 0.0
    best = 0.0
    for stored in gallery:
        best = max(best, _cosine_similarity(query, stored))
    return best


def _gallery_pair_similarity(
    left: deque[np.ndarray], right: deque[np.ndarray]
) -> float:
    """Return the best normalized similarity between two identity galleries."""
    if not left or not right:
        return 0.0
    return max(
        _cosine_similarity(first, second)
        for first in left
        for second in right
    )


def _same_camera_duplicate(left: IdentityRecord, right: IdentityRecord) -> bool:
    """Whether two same-camera records look like overlapping duplicate boxes."""
    if left.bbox is None or right.bbox is None:
        return False
    intersection = _bbox_intersection(left.bbox, right.bbox)
    if intersection <= 0:
        return False
    smaller_area = min(left.bbox.area, right.bbox.area)
    union = left.bbox.area + right.bbox.area - intersection
    if smaller_area <= 0 or union <= 0:
        return False
    containment = intersection / smaller_area
    iou = intersection / union
    return containment >= 0.85 or iou >= 0.50


def _bbox_intersection(left: BoundingBox, right: BoundingBox) -> float:
    return max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1)) * max(
        0.0, min(left.y2, right.y2) - max(left.y1, right.y1)
    )


def _bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    intersection = _bbox_intersection(left, right)
    if intersection <= 0:
        return 0.0
    union = left.area + right.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _normalize_embedding(embedding: np.ndarray | None) -> np.ndarray | None:
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else None


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
    "CameraSighting",
    "EmbeddingExtractor",
    "GlobalIdentityConfig",
    "GlobalIdentityManager",
    "IdentityRecord",
    "IdentityState",
]
