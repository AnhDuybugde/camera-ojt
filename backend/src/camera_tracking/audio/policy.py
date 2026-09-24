"""Priority, cooldown, dwell and privacy policy for semantic audio events."""

from __future__ import annotations

from dataclasses import dataclass
import os

from camera_tracking.audio.events import AudioEventKind


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class EventPolicy:
    priority: int
    cooldown_s: float
    expires_s: float
    allow_name: bool = True
    generic_zone_min_dwell_s: float = 0.0


@dataclass(frozen=True, slots=True)
class AudioPolicy:
    enabled: bool
    global_gap_s: float
    same_person_gap_s: float
    max_event_age_s: float
    future_tolerance_s: float
    dedupe_ttl_s: float
    dedupe_max_entries: int
    group_arrivals: bool
    policies: dict[AudioEventKind, EventPolicy]

    def for_kind(self, kind: AudioEventKind) -> EventPolicy:
        return self.policies[kind]

    @classmethod
    def from_env(cls) -> "AudioPolicy":
        return cls(
            enabled=_env_bool("HAMY_ZONE_EVENTS_ENABLED", True),
            global_gap_s=_env_float("HAMY_EVENT_GLOBAL_GAP_SECONDS", 3.0),
            same_person_gap_s=_env_float("HAMY_EVENT_PERSON_GAP_SECONDS", 8.0),
            max_event_age_s=_env_float("HAMY_EVENT_MAX_AGE_SECONDS", 20.0, 1.0),
            future_tolerance_s=_env_float("HAMY_EVENT_FUTURE_TOLERANCE_SECONDS", 10.0),
            dedupe_ttl_s=_env_float("HAMY_EVENT_DEDUPE_TTL_SECONDS", 3600.0, 30.0),
            dedupe_max_entries=_env_int("HAMY_EVENT_DEDUPE_MAX_ENTRIES", 4096, 128),
            group_arrivals=_env_bool("HAMY_GROUP_DOOR_ARRIVALS", True),
            policies={
                AudioEventKind.WAVE: EventPolicy(
                    priority=0,
                    cooldown_s=_env_float("HAMY_WAVE_COOLDOWN_SECONDS", 10.0),
                    expires_s=8.0,
                ),
                AudioEventKind.DOOR_ENTER: EventPolicy(
                    priority=10,
                    cooldown_s=_env_float("HAMY_DOOR_ENTER_COOLDOWN_SECONDS", 300.0),
                    expires_s=8.0,
                ),
                AudioEventKind.DOOR_EXIT: EventPolicy(
                    priority=12,
                    cooldown_s=_env_float("HAMY_DOOR_EXIT_COOLDOWN_SECONDS", 300.0),
                    expires_s=8.0,
                ),
                AudioEventKind.BE_XINH_NEAR: EventPolicy(
                    priority=20,
                    cooldown_s=_env_float("HAMY_BE_XINH_COOLDOWN_SECONDS", 600.0),
                    expires_s=10.0,
                    generic_zone_min_dwell_s=_env_float("HAMY_BE_XINH_MIN_DWELL_SECONDS", 0.8),
                ),
                AudioEventKind.WATER: EventPolicy(
                    priority=40,
                    cooldown_s=_env_float("HAMY_WATER_ZONE_COOLDOWN_SECONDS", 3600.0),
                    expires_s=20.0,
                    generic_zone_min_dwell_s=_env_float("HAMY_WATER_ZONE_MIN_DWELL_SECONDS", 1.5),
                ),
                AudioEventKind.RESTROOM: EventPolicy(
                    priority=50,
                    cooldown_s=_env_float("HAMY_RESTROOM_COOLDOWN_SECONDS", 1800.0),
                    expires_s=10.0,
                    allow_name=False,
                    generic_zone_min_dwell_s=_env_float("HAMY_RESTROOM_MIN_DWELL_SECONDS", 1.0),
                ),
            },
        )


__all__ = ["AudioPolicy", "EventPolicy"]
