"""Central configuration for lightweight, rule-based activity estimation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class ActivityConfig:
    enabled: bool = field(default_factory=lambda: _bool("ACTIVITY_ENABLED", True))
    pose_model: str = field(default_factory=lambda: os.getenv("POSE_MODEL", "yolo11n-pose.pt").strip())
    process_every_n_frames: int = field(default_factory=lambda: _int("ACTIVITY_PROCESS_EVERY_N_FRAMES", 3))
    pose_image_size: int = field(default_factory=lambda: _int("ACTIVITY_POSE_IMAGE_SIZE", 416, 256))
    pose_confidence: float = field(default_factory=lambda: _float("ACTIVITY_POSE_CONFIDENCE", 0.35))
    window_seconds: float = field(default_factory=lambda: _float("ACTIVITY_WINDOW_SECONDS", 5.0, 1.0))
    confirm_ratio: float = field(default_factory=lambda: min(1.0, _float("ACTIVITY_CONFIRM_RATIO", 0.70)))
    sleep_min_seconds: float = field(default_factory=lambda: _float("ACTIVITY_SLEEP_MIN_SECONDS", 25.0, 3.0))
    drink_min_seconds: float = field(default_factory=lambda: _float("ACTIVITY_DRINK_MIN_SECONDS", 1.5, 0.5))
    walking_min_seconds: float = field(default_factory=lambda: _float("ACTIVITY_WALKING_MIN_SECONDS", 1.2, 0.3))
    standing_min_seconds: float = field(default_factory=lambda: _float("ACTIVITY_STANDING_MIN_SECONDS", 1.5, 0.3))
    desk_min_seconds: float = field(default_factory=lambda: _float("ACTIVITY_DESK_MIN_SECONDS", 2.0, 0.5))
    away_min_seconds: float = field(default_factory=lambda: _float("ACTIVITY_AWAY_MIN_SECONDS", 2.0, 0.5))
    movement_threshold: float = field(default_factory=lambda: _float("ACTIVITY_MOVEMENT_THRESHOLD", 0.035, 0.005))
    identity_ttl_seconds: float = field(default_factory=lambda: _float("ACTIVITY_IDENTITY_TTL_SECONDS", 15.0, 2.0))
    track_max_distance: float = field(default_factory=lambda: _float("ACTIVITY_TRACK_DISTANCE", 140.0, 20.0))
    track_max_missed: int = field(default_factory=lambda: _int("ACTIVITY_TRACK_MAX_MISSED", 8))
    desk_zone_ids: tuple[str, ...] = field(default_factory=lambda: tuple(
        item.strip().lower() for item in os.getenv(
            "ACTIVITY_DESK_ZONE_IDS", "workspace,desk,desk_zone_1"
        ).split(",") if item.strip()
    ))

    def minimum_duration(self, activity: str) -> float:
        return {
            "WALKING": self.walking_min_seconds,
            "STANDING": self.standing_min_seconds,
            "DRINKING": self.drink_min_seconds,
            "SLEEPING_SUSPECTED": self.sleep_min_seconds,
            "AT_DESK": self.desk_min_seconds,
            "AWAY": self.away_min_seconds,
            "UNKNOWN": 0.0,
        }.get(activity, 0.0)


activity_settings = ActivityConfig()
