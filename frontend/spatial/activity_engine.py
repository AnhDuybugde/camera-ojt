"""Fail-safe activity worker for live cameras; never mutates attendance state."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import cv2
import numpy as np

from camera.camera_manager import CameraManager
from spatial.activity_config import ActivityConfig
from spatial.activity_rules import (
    ACTIVITY_COLORS, ACTIVITY_LABELS, ActivityType, TemporalActivityState,
    classify_activity, smooth_activity,
)
from spatial.pose_service import PoseObservation, PoseService
from spatial.repository import SpatialRepository, local_now
from spatial.tracker import CentroidTracker, TrackSnapshot
from spatial.zones import Zone, zone_at

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IdentityObservation:
    box: tuple[int, int, int, int]
    employee_id: str
    employee_name: str


@dataclass(frozen=True, slots=True)
class IdentityBinding:
    employee_id: str
    employee_name: str
    expires_at: float


@dataclass(frozen=True, slots=True)
class ActivityView:
    camera_id: str
    track_id: int
    box: tuple[int, int, int, int]
    employee_id: str | None
    employee_name: str
    activity: ActivityType
    label: str
    confidence: float
    zone: str | None
    started_at: datetime
    duration_seconds: float


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / max(1, left_area + right_area - intersection)


class ActivityRuntime:
    """One controlled worker per camera with shared, reusable PoseService."""

    def __init__(
        self, camera_id: str, camera: CameraManager, pose: PoseService,
        repository: SpatialRepository, zones: tuple[Zone, ...], config: ActivityConfig,
    ) -> None:
        self.camera_id, self.camera, self.pose = camera_id, camera, pose
        self.repository, self.zones, self.config = repository, zones, config
        self.tracker = CentroidTracker(
            max_distance=config.track_max_distance, max_missed=config.track_max_missed,
        )
        self._states: dict[int, TemporalActivityState] = {}
        self._identities: dict[int, IdentityBinding] = {}
        self._pending_identities: list[IdentityObservation] = []
        self._views: list[ActivityView] = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._processed_frames = 0
        self._activity_times: deque[float] = deque(maxlen=30)
        self._started_mono = 0.0
        self.last_pose_ms = 0
        self.activity_fps = 0.0
        self.error = ""

    def start(self) -> "ActivityRuntime":
        if not self.config.enabled or (self._thread and self._thread.is_alive()):
            return self
        self._stop.clear()
        self._started_mono = time.monotonic()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"activity-{self.camera_id}",
        )
        self._thread.start()
        return self

    def submit_identities(self, observations: list[IdentityObservation]) -> None:
        with self._lock:
            self._pending_identities = list(observations)

    def _loop(self) -> None:
        frame_index = 0
        last_sequence = -1
        retry_after = 0.0
        while not self._stop.is_set():
            ok, frame = self.camera.read()
            if not ok or frame is None:
                self._stop.wait(0.08)
                continue
            sequence = self.camera.frame_sequence
            if sequence == last_sequence:
                self._stop.wait(0.02)
                continue
            last_sequence = sequence
            frame_index += 1
            if frame_index % self.config.process_every_n_frames:
                self._stop.wait(0.04)
                continue
            now_mono = time.monotonic()
            if now_mono < retry_after:
                self._stop.wait(min(0.2, retry_after - now_mono))
                continue
            started = time.monotonic()
            try:
                poses = self.pose.infer(frame)
                self.last_pose_ms = int((time.monotonic() - started) * 1000)
                self._process(poses, frame.shape[1], frame.shape[0], now_mono)
                self._processed_frames += 1
                completed_at = time.monotonic()
                self._activity_times.append(completed_at)
                if len(self._activity_times) >= 2:
                    elapsed = self._activity_times[-1] - self._activity_times[0]
                    self.activity_fps = (len(self._activity_times) - 1) / max(0.001, elapsed)
                self.error = ""
            except Exception as exc:
                # Attendance owns a separate worker; activity failure is isolated here.
                self.error = f"Activity tạm dừng: {exc}"
                retry_after = time.monotonic() + 5.0
                logger.exception("Activity worker failed for %s", self.camera_id)
            self._stop.wait(0.04)

    def _process(
        self, poses: list[PoseObservation], width: int, height: int, now_mono: float,
    ) -> None:
        with self._lock:
            tracks = self.tracker.update([pose.box for pose in poses])
            self._bind_pending(tracks, now_mono)
            views: list[ActivityView] = []
            now_wall = local_now()
            for track in tracks:
                binding = self._identities.get(track.track_id)
                if binding and binding.expires_at < now_mono:
                    self._identities.pop(track.track_id, None)
                    binding = None
                pose = max(poses, key=lambda item: _iou(track.box, item.box), default=None)
                if pose is not None and _iou(track.box, pose.box) < 0.15:
                    pose = None
                zone = zone_at(self.zones, track.anchor[0] / width, track.anchor[1] / height)
                zone_id = zone.zone_id if zone else None
                state = self._states.setdefault(track.track_id, TemporalActivityState())
                in_desk = bool(zone_id and zone_id.lower() in self.config.desk_zone_ids)
                result = classify_activity(
                    track, pose, frame_width=width, frame_height=height,
                    in_desk=in_desk, was_at_desk=state.was_at_desk, config=self.config,
                )
                state.was_at_desk = state.was_at_desk or in_desk
                changed = smooth_activity(state, result, now_mono, self.config)
                if changed:
                    self._persist_transition(track, binding, state, zone_id, now_wall)
                employee_id = binding.employee_id if binding else None
                employee_name = binding.employee_name if binding else "Chưa xác định"
                duration = max(0.0, now_mono - state.started_at)
                started_at = now_wall - timedelta(seconds=duration)
                views.append(ActivityView(
                    self.camera_id, track.track_id, track.box, employee_id, employee_name,
                    state.current, ACTIVITY_LABELS[state.current], state.confidence, zone_id,
                    started_at, duration,
                ))
            for retired in self.tracker.pop_retired():
                self._close_track(retired.track_id, now_wall)
            self._views = views

    def _bind_pending(self, tracks: list[TrackSnapshot], now_mono: float) -> None:
        observations, self._pending_identities = self._pending_identities, []
        for identity in observations:
            center = ((identity.box[0] + identity.box[2]) / 2, (identity.box[1] + identity.box[3]) / 2)
            candidates = [track for track in tracks if (
                track.box[0] <= center[0] <= track.box[2]
                and track.box[1] <= center[1] <= track.box[3]
            )]
            if not candidates:
                continue
            track = min(candidates, key=lambda item: abs(item.anchor[0] - center[0]))
            previous = self._identities.get(track.track_id)
            self._identities[track.track_id] = IdentityBinding(
                identity.employee_id, identity.employee_name,
                now_mono + self.config.identity_ttl_seconds,
            )
            self.tracker.identify(track.track_id, identity.employee_id, identity.employee_name)
            if previous is None or previous.employee_id != identity.employee_id:
                try:
                    self.repository.bind_activity_identity(
                        self.camera_id, track.track_id, identity.employee_id, identity.employee_name,
                    )
                except Exception as exc:
                    logger.warning("Could not bind activity identity: %s", exc)

    def _persist_transition(
        self, track: TrackSnapshot, binding: IdentityBinding | None,
        state: TemporalActivityState, zone_id: str | None, at: datetime,
    ) -> None:
        try:
            self.repository.transition_activity(
                self.camera_id, track.track_id,
                binding.employee_id if binding else None,
                binding.employee_name if binding else "Chưa xác định",
                state.current.value, state.confidence, zone_id, at,
            )
        except Exception as exc:
            logger.warning("Activity event persistence failed: %s", exc)

    def _close_track(self, track_id: int, at: datetime) -> None:
        try:
            self.repository.close_activity(self.camera_id, track_id, at)
        except Exception as exc:
            logger.warning("Could not close activity event: %s", exc)
        self._states.pop(track_id, None)
        self._identities.pop(track_id, None)

    def latest(self) -> list[ActivityView]:
        with self._lock:
            return list(self._views)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        output = frame.copy()
        for item in self.latest():
            x1, y1, x2, y2 = item.box
            color = ACTIVITY_COLORS[item.activity]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            identity = item.employee_id or f"Track #{item.track_id}"
            label = f"{identity} | {item.activity.value} {item.confidence:.0%}"
            cv2.putText(output, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
        return output

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=4)
        now = local_now()
        for track_id in list(self._states):
            self._close_track(track_id, now)
