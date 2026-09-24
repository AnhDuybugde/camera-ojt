"""Observable pose/spatial rules and temporal smoothing for employee activity."""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import StrEnum

from spatial.activity_config import ActivityConfig
from spatial.pose_service import PoseObservation
from spatial.tracker import TrackSnapshot


class ActivityType(StrEnum):
    AT_DESK = "AT_DESK"
    WALKING = "WALKING"
    STANDING = "STANDING"
    DRINKING = "DRINKING"
    SLEEPING_SUSPECTED = "SLEEPING_SUSPECTED"
    AWAY = "AWAY"
    UNKNOWN = "UNKNOWN"


ACTIVITY_LABELS = {
    ActivityType.AT_DESK: "Tại bàn",
    ActivityType.WALKING: "Đang di chuyển",
    ActivityType.STANDING: "Đang đứng",
    ActivityType.DRINKING: "Đang uống nước",
    ActivityType.SLEEPING_SUSPECTED: "Có dấu hiệu ngủ",
    ActivityType.AWAY: "Rời vị trí",
    ActivityType.UNKNOWN: "Không xác định",
}

ACTIVITY_COLORS = {
    ActivityType.AT_DESK: (16, 185, 129),
    ActivityType.WALKING: (235, 158, 52),
    ActivityType.STANDING: (148, 163, 184),
    ActivityType.DRINKING: (207, 191, 6),
    ActivityType.SLEEPING_SUSPECTED: (14, 165, 233),
    ActivityType.AWAY: (11, 158, 245),
    ActivityType.UNKNOWN: (107, 114, 128),
}


@dataclass(frozen=True, slots=True)
class RuleResult:
    activity: ActivityType
    confidence: float


@dataclass(slots=True)
class TemporalActivityState:
    current: ActivityType = ActivityType.UNKNOWN
    confidence: float = 0.0
    started_at: float = 0.0
    candidate: ActivityType = ActivityType.UNKNOWN
    candidate_since: float = 0.0
    history: deque[tuple[float, ActivityType, float]] = field(default_factory=deque)
    was_at_desk: bool = False


def _point(pose: PoseObservation, name: str, minimum: float = 0.25):
    value = pose.keypoints.get(name)
    return value if value and value[2] >= minimum else None


def _distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def movement_ratio(track: TrackSnapshot, frame_width: int, frame_height: int) -> float:
    if len(track.trail) < 2:
        return 0.0
    diagonal = max(1.0, math.hypot(frame_width, frame_height))
    # Net displacement prevents small detection jitter from becoming WALKING.
    return _distance(track.trail[0], track.trail[-1]) / diagonal


def drinking_score(pose: PoseObservation | None) -> float:
    if pose is None:
        return 0.0
    nose = _point(pose, "nose")
    shoulders = [_point(pose, name) for name in ("left_shoulder", "right_shoulder")]
    shoulder = next((item for item in shoulders if item), None)
    if not nose or not shoulder:
        return 0.0
    torso = max(20.0, abs(shoulder[1] - pose.box[3]) * 0.45)
    scores = []
    for side in ("left", "right"):
        wrist, elbow = _point(pose, f"{side}_wrist"), _point(pose, f"{side}_elbow")
        if wrist and elbow:
            near_face = _distance(wrist, nose) <= torso * 0.55
            elbow_raised = elbow[1] <= shoulder[1] + torso * 0.55
            if near_face and elbow_raised:
                scores.append(min(0.96, 0.70 + (wrist[2] + elbow[2]) * 0.13))
    return max(scores, default=0.0)


def sleeping_score(pose: PoseObservation | None, movement: float, in_desk: bool) -> float:
    if pose is None or not in_desk:
        return 0.0
    nose = _point(pose, "nose")
    left_shoulder, right_shoulder = _point(pose, "left_shoulder"), _point(pose, "right_shoulder")
    shoulders = [p for p in (left_shoulder, right_shoulder) if p]
    hips = [p for p in (_point(pose, "left_hip"), _point(pose, "right_hip")) if p]
    if not nose or not shoulders or not hips or movement > 0.012:
        return 0.0
    shoulder_y = sum(p[1] for p in shoulders) / len(shoulders)
    hip_y = sum(p[1] for p in hips) / len(hips)
    torso = max(10.0, hip_y - shoulder_y)
    # Nose at/below the shoulder line is a conservative head-down signal.
    drop = (nose[1] - shoulder_y) / torso
    return min(0.92, 0.68 + max(0.0, drop) * 0.25) if drop >= -0.05 else 0.0


def standing_score(pose: PoseObservation | None) -> float:
    if pose is None:
        return 0.0
    shoulders = [p for p in (_point(pose, "left_shoulder"), _point(pose, "right_shoulder")) if p]
    hips = [p for p in (_point(pose, "left_hip"), _point(pose, "right_hip")) if p]
    knees = [p for p in (_point(pose, "left_knee"), _point(pose, "right_knee")) if p]
    ankles = [p for p in (_point(pose, "left_ankle"), _point(pose, "right_ankle")) if p]
    if not shoulders or not hips or not knees:
        return 0.0
    sy = sum(p[1] for p in shoulders) / len(shoulders)
    hy = sum(p[1] for p in hips) / len(hips)
    ky = sum(p[1] for p in knees) / len(knees)
    vertical = sy < hy < ky
    extended = not ankles or ky < sum(p[1] for p in ankles) / len(ankles)
    return 0.78 if vertical and extended else 0.0


def classify_activity(
    track: TrackSnapshot, pose: PoseObservation | None, *, frame_width: int,
    frame_height: int, in_desk: bool, was_at_desk: bool,
    config: ActivityConfig,
) -> RuleResult:
    movement = movement_ratio(track, frame_width, frame_height)
    drink = drinking_score(pose)
    sleep = sleeping_score(pose, movement, in_desk)
    standing = standing_score(pose)
    # Priority deliberately allows special observable actions to override zone state.
    if drink:
        return RuleResult(ActivityType.DRINKING, drink)
    if sleep:
        return RuleResult(ActivityType.SLEEPING_SUSPECTED, sleep)
    if movement >= config.movement_threshold:
        return RuleResult(ActivityType.WALKING, min(0.95, 0.70 + movement * 2.5))
    if was_at_desk and not in_desk:
        return RuleResult(ActivityType.AWAY, 0.76)
    if standing:
        return RuleResult(ActivityType.STANDING, standing)
    if in_desk:
        return RuleResult(ActivityType.AT_DESK, 0.82 if pose else 0.72)
    return RuleResult(ActivityType.UNKNOWN, 0.35)


def smooth_activity(
    state: TemporalActivityState, result: RuleResult, now: float,
    config: ActivityConfig,
) -> bool:
    """Update state and return True only when a confirmed transition occurs."""
    if state.started_at == 0.0:
        state.started_at = now
    state.history.append((now, result.activity, result.confidence))
    cutoff = now - config.window_seconds
    while state.history and state.history[0][0] < cutoff:
        state.history.popleft()
    if result.activity != state.candidate:
        state.candidate = result.activity
        state.candidate_since = now

    counts = Counter(item[1] for item in state.history)
    ratio = counts[result.activity] / max(1, len(state.history))
    duration = now - state.candidate_since
    if (result.activity != state.current
            and ratio >= config.confirm_ratio
            and duration >= config.minimum_duration(result.activity)):
        relevant = [item[2] for item in state.history if item[1] == result.activity]
        state.current = result.activity
        state.confidence = sum(relevant) / max(1, len(relevant))
        state.started_at = now
        return True
    if result.activity == state.current:
        state.confidence = state.confidence * 0.7 + result.confidence * 0.3
    return False
