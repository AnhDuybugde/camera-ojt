"""Very lightweight motion-event analysis for the Bé Xinh companion.

This module deliberately avoids another neural network. It derives interaction
signals from the YOLO/ByteTrack boxes already produced by Camera AIM:

* APPROACH: a tracked person moves toward the camera and grows in frame.
* STAND_UP: a previously stable/seated box becomes markedly taller while its
  floor contact stays near the same place.
* STATIONARY: used only for long water/rest reminders.
* PROXIMITY: an approximate monocular distance derived from calibrated person
  box height, with confirmation and hysteresis to avoid boundary chatter.

The calculations are O(number_of_tracks) and do not add GPU inference.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import statistics
from typing import Iterable

from camera_tracking.domain import Track


@dataclass(frozen=True)
class InteractionEvent:
    kind: str
    channel: str
    global_id: int
    now_s: float
    confidence: float = 1.0


@dataclass(frozen=True)
class MotionSnapshot:
    channel: str
    global_id: int
    first_seen_s: float
    last_seen_s: float
    stationary: bool
    stationary_since_s: float | None
    area_ratio: float
    center_x: float
    center_y: float
    box_height_ratio: float
    near_camera: bool
    estimated_distance_m: float | None


@dataclass(frozen=True)
class _Sample:
    t: float
    cx: float
    cy: float
    foot_x: float
    foot_y: float
    area: float
    height: float
    top_y: float


@dataclass
class _TrackMotionState:
    first_seen_s: float
    last_seen_s: float
    samples: deque[_Sample] = field(default_factory=lambda: deque(maxlen=80))
    stationary_since_s: float | None = None
    last_approach_s: float = float("-inf")
    last_stand_s: float = float("-inf")
    near_started_s: float | None = None
    near_camera: bool = False
    estimated_distance_m: float | None = None


class InteractionEngine:
    """Turn existing tracking boxes into sparse, high-value companion events."""

    def __init__(
        self,
        *,
        stationary_window_s: float = 4.0,
        stationary_center_delta: float = 0.025,
        stationary_area_delta: float = 0.20,
        approach_lookback_s: float = 1.2,
        approach_area_growth: float = 1.45,
        approach_min_area_ratio: float = 0.025,
        approach_motion: float = 0.035,
        approach_cooldown_s: float = 90.0,
        stand_height_growth: float = 1.25,
        stand_top_rise: float = 0.045,
        stand_foot_delta: float = 0.08,
        stand_cooldown_s: float = 120.0,
        retire_after_s: float = 20.0,
        distance_reference_m: float = 1.0,
        distance_reference_height_ratio: float = 0.45,
        greeting_distance_m: float = 0.50,
        distance_release_m: float = 0.70,
        near_confirm_s: float = 0.25,
    ) -> None:
        self.stationary_window_s = max(2.0, stationary_window_s)
        self.stationary_center_delta = max(0.005, stationary_center_delta)
        self.stationary_area_delta = max(0.05, stationary_area_delta)
        self.approach_lookback_s = max(0.5, approach_lookback_s)
        self.approach_area_growth = max(1.1, approach_area_growth)
        self.approach_min_area_ratio = max(0.005, approach_min_area_ratio)
        self.approach_motion = max(0.005, approach_motion)
        self.approach_cooldown_s = max(10.0, approach_cooldown_s)
        self.stand_height_growth = max(1.05, stand_height_growth)
        self.stand_top_rise = max(0.01, stand_top_rise)
        self.stand_foot_delta = max(0.02, stand_foot_delta)
        self.stand_cooldown_s = max(10.0, stand_cooldown_s)
        self.retire_after_s = max(5.0, retire_after_s)
        self.distance_reference_m = max(0.10, distance_reference_m)
        self.distance_reference_height_ratio = max(
            0.02, distance_reference_height_ratio
        )
        self.greeting_distance_m = max(0.20, greeting_distance_m)
        self.distance_release_m = max(
            self.greeting_distance_m + 0.05, distance_release_m
        )
        self.near_confirm_s = max(0.0, near_confirm_s)
        self._states: dict[tuple[str, int], _TrackMotionState] = {}
        self._snapshots: dict[tuple[str, int], MotionSnapshot] = {}

    def observe_tracks(
        self,
        channel: str,
        tracks: Iterable[Track],
        frame_shape: tuple[int, ...],
        now_s: float,
    ) -> list[InteractionEvent]:
        height_px = max(1, int(frame_shape[0]))
        width_px = max(1, int(frame_shape[1]))
        frame_area = float(width_px * height_px)
        events: list[InteractionEvent] = []
        seen_keys: set[tuple[str, int]] = set()

        for track in tracks:
            gid = int(track.global_person_id or track.track_id)
            key = (channel, gid)
            seen_keys.add(key)
            bbox = track.bbox
            cx = ((bbox.x1 + bbox.x2) * 0.5) / width_px
            cy = ((bbox.y1 + bbox.y2) * 0.5) / height_px
            foot_x = cx
            foot_y = bbox.y2 / height_px
            area = max(1.0, bbox.area) / frame_area
            box_height = max(1.0, bbox.height) / height_px
            top_y = bbox.y1 / height_px
            sample = _Sample(
                t=now_s,
                cx=cx,
                cy=cy,
                foot_x=foot_x,
                foot_y=foot_y,
                area=area,
                height=box_height,
                top_y=top_y,
            )

            state = self._states.get(key)
            if state is None:
                state = _TrackMotionState(first_seen_s=now_s, last_seen_s=now_s)
                state.samples.append(sample)
                self._update_proximity(state, sample, now_s)
                self._states[key] = state
                self._set_snapshot(channel, gid, state, sample, stationary=False)
                continue

            state.last_seen_s = now_s
            old_for_approach = self._sample_before(
                state.samples, now_s - self.approach_lookback_s
            )
            stationary_reference = self._sample_before(
                state.samples, now_s - self.stationary_window_s
            )
            was_stationary_since = state.stationary_since_s

            # Stand-up is checked against the stable history BEFORE the current
            # movement is allowed to reset stationary_since_s.
            if (
                was_stationary_since is not None
                and now_s - was_stationary_since >= 3.0
                and now_s - state.last_stand_s >= self.stand_cooldown_s
                and len(state.samples) >= 6
            ):
                historical = [
                    item
                    for item in state.samples
                    if now_s - 8.0 <= item.t <= now_s - 0.8
                ]
                if len(historical) >= 4:
                    base_h = statistics.median(item.height for item in historical)
                    base_top = statistics.median(item.top_y for item in historical)
                    base_foot_y = statistics.median(item.foot_y for item in historical)
                    height_growth = sample.height / max(base_h, 1e-6)
                    top_rise = base_top - sample.top_y
                    foot_delta = abs(sample.foot_y - base_foot_y)
                    if (
                        height_growth >= self.stand_height_growth
                        and top_rise >= self.stand_top_rise
                        and foot_delta <= self.stand_foot_delta
                    ):
                        confidence = min(
                            1.0,
                            0.55
                            + (height_growth - self.stand_height_growth) * 0.8
                            + top_rise * 2.0,
                        )
                        events.append(
                            InteractionEvent(
                                "STAND_UP", channel, gid, now_s, confidence
                            )
                        )
                        state.last_stand_s = now_s

            if (
                old_for_approach is not None
                and now_s - state.last_approach_s >= self.approach_cooldown_s
            ):
                area_growth = sample.area / max(old_for_approach.area, 1e-6)
                motion = math.hypot(
                    sample.cx - old_for_approach.cx,
                    sample.cy - old_for_approach.cy,
                )
                if (
                    sample.area >= self.approach_min_area_ratio
                    and area_growth >= self.approach_area_growth
                    and motion >= self.approach_motion
                ):
                    confidence = min(
                        1.0,
                        0.50
                        + (area_growth - self.approach_area_growth) * 0.5
                        + motion * 2.0,
                    )
                    events.append(
                        InteractionEvent("APPROACH", channel, gid, now_s, confidence)
                    )
                    state.last_approach_s = now_s

            stationary = False
            if stationary_reference is not None:
                center_delta = math.hypot(
                    sample.cx - stationary_reference.cx,
                    sample.cy - stationary_reference.cy,
                )
                area_delta = abs(sample.area - stationary_reference.area) / max(
                    stationary_reference.area, 1e-6
                )
                stationary = (
                    center_delta <= self.stationary_center_delta
                    and area_delta <= self.stationary_area_delta
                )

            if stationary:
                if state.stationary_since_s is None:
                    state.stationary_since_s = max(
                        state.first_seen_s, now_s - self.stationary_window_s
                    )
            else:
                state.stationary_since_s = None

            state.samples.append(sample)
            self._update_proximity(state, sample, now_s)
            self._set_snapshot(channel, gid, state, sample, stationary=stationary)

        self._retire(now_s)
        return events

    def snapshot(self, channel: str, global_id: int) -> MotionSnapshot | None:
        return self._snapshots.get((channel, int(global_id)))

    def snapshots(self) -> list[MotionSnapshot]:
        return list(self._snapshots.values())

    def forget_global_ids(self, alive_global_ids: set[int]) -> None:
        for key in [key for key in self._states if key[1] not in alive_global_ids]:
            self._states.pop(key, None)
            self._snapshots.pop(key, None)

    @staticmethod
    def _sample_before(samples: deque[_Sample], target_s: float) -> _Sample | None:
        candidate = None
        for item in samples:
            if item.t <= target_s:
                candidate = item
            else:
                break
        return candidate

    def _set_snapshot(
        self,
        channel: str,
        gid: int,
        state: _TrackMotionState,
        sample: _Sample,
        *,
        stationary: bool,
    ) -> None:
        self._snapshots[(channel, gid)] = MotionSnapshot(
            channel=channel,
            global_id=gid,
            first_seen_s=state.first_seen_s,
            last_seen_s=state.last_seen_s,
            stationary=stationary,
            stationary_since_s=state.stationary_since_s,
            area_ratio=sample.area,
            center_x=sample.cx,
            center_y=sample.cy,
            box_height_ratio=sample.height,
            near_camera=state.near_camera,
            estimated_distance_m=state.estimated_distance_m,
        )

    def _update_proximity(
        self,
        state: _TrackMotionState,
        sample: _Sample,
        now_s: float,
    ) -> None:
        """Update a calibrated monocular distance proxy.

        ``distance_reference_height_ratio`` must be measured once by asking a
        standing person to occupy a known reference distance. This is not a
        depth sensor, so the estimate is deliberately exposed as approximate.
        """
        distance_m = (
            self.distance_reference_m
            * self.distance_reference_height_ratio
            / max(sample.height, 1e-6)
        )
        state.estimated_distance_m = distance_m

        if state.near_camera:
            if distance_m >= self.distance_release_m:
                state.near_camera = False
                state.near_started_s = None
            return

        if distance_m > self.greeting_distance_m:
            state.near_started_s = None
            return
        if state.near_started_s is None:
            state.near_started_s = now_s
            return
        if now_s - state.near_started_s >= self.near_confirm_s:
            state.near_camera = True

    def _retire(self, now_s: float) -> None:
        stale = [
            key
            for key, state in self._states.items()
            if now_s - state.last_seen_s >= self.retire_after_s
        ]
        for key in stale:
            self._states.pop(key, None)
            self._snapshots.pop(key, None)


__all__ = ["InteractionEngine", "InteractionEvent", "MotionSnapshot"]
