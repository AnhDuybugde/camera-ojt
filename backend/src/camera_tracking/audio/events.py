"""Validated semantic events at the tracking-to-audio boundary.

Detection owns geometry, direction and dwell. Audio only consumes the small,
stable event vocabulary defined here and never reopens camera streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping


class AudioEventKind(str, Enum):
    DOOR_ENTER = "door_enter"
    DOOR_EXIT = "door_exit"
    BE_XINH_NEAR = "be_xinh_near"
    WATER = "water"
    RESTROOM = "restroom"
    WAVE = "wave"


_ZONE_ALIASES = {
    "door_inside": "door_inside",
    "inside_door": "door_inside",
    "entrance_inside": "door_inside",
    "door_in": "door_inside",
    "door_outside": "door_outside",
    "outside_door": "door_outside",
    "entrance_outside": "door_outside",
    "door_out": "door_outside",
    "be_xinh": "be_xinh",
    "bexinh": "be_xinh",
    "be_xinh_zone": "be_xinh",
    "hamy": "be_xinh",
    "water": "water",
    "water_zone": "water",
    "drinking_water": "water",
    "pantry_water": "water",
    "restroom": "restroom",
    "toilet": "restroom",
    "wc": "restroom",
    "bathroom": "restroom",
}

_EVENT_ALIASES = {
    "door_enter": AudioEventKind.DOOR_ENTER,
    "enter_office": AudioEventKind.DOOR_ENTER,
    "office_enter": AudioEventKind.DOOR_ENTER,
    "arrival": AudioEventKind.DOOR_ENTER,
    "door_exit": AudioEventKind.DOOR_EXIT,
    "exit_office": AudioEventKind.DOOR_EXIT,
    "office_exit": AudioEventKind.DOOR_EXIT,
    "leaving": AudioEventKind.DOOR_EXIT,
    "be_xinh_near": AudioEventKind.BE_XINH_NEAR,
    "bexinh_near": AudioEventKind.BE_XINH_NEAR,
    "near_be_xinh": AudioEventKind.BE_XINH_NEAR,
    "approach_be_xinh": AudioEventKind.BE_XINH_NEAR,
    "water": AudioEventKind.WATER,
    "water_zone": AudioEventKind.WATER,
    "drink_water": AudioEventKind.WATER,
    "restroom": AudioEventKind.RESTROOM,
    "toilet": AudioEventKind.RESTROOM,
    "wc": AudioEventKind.RESTROOM,
    "wave": AudioEventKind.WAVE,
    "hand_wave": AudioEventKind.WAVE,
}

_GENERIC_ZONE_TYPES = {"zone_enter", "zone_exit", "zone_dwell", "zone_cross"}


def _clean_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _clean_text(value: Any, max_length: int = 200) -> str:
    return " ".join(str(value or "").split())[:max_length]


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def canonical_event_kind(
    raw_type: Any,
    *,
    zone: Any = None,
    direction: Any = None,
) -> AudioEventKind | None:
    """Map semantic or generic zone events to the stable audio vocabulary."""

    event_type = _clean_token(raw_type)
    if event_type in _EVENT_ALIASES:
        return _EVENT_ALIASES[event_type]
    if event_type not in _GENERIC_ZONE_TYPES:
        return None

    zone_name = _ZONE_ALIASES.get(_clean_token(zone), _clean_token(zone))
    direction_name = _clean_token(direction)
    if zone_name == "door_inside":
        if direction_name == "inside_to_outside":
            return AudioEventKind.DOOR_EXIT
        if direction_name == "outside_to_inside":
            return AudioEventKind.DOOR_ENTER
        if event_type == "zone_enter":
            return AudioEventKind.DOOR_ENTER
        if event_type == "zone_exit":
            return AudioEventKind.DOOR_EXIT
        return None
    if zone_name == "door_outside":
        if direction_name == "inside_to_outside":
            return AudioEventKind.DOOR_EXIT
        if direction_name == "outside_to_inside":
            return AudioEventKind.DOOR_ENTER
        if event_type == "zone_enter":
            return AudioEventKind.DOOR_EXIT
        if event_type == "zone_exit":
            return AudioEventKind.DOOR_ENTER
        return None
    if zone_name == "be_xinh" and event_type in {"zone_enter", "zone_dwell"}:
        return AudioEventKind.BE_XINH_NEAR
    if zone_name == "water" and event_type in {"zone_enter", "zone_dwell"}:
        return AudioEventKind.WATER
    if zone_name == "restroom" and event_type in {"zone_enter", "zone_dwell"}:
        return AudioEventKind.RESTROOM
    return None


@dataclass(frozen=True, slots=True)
class AudioEvent:
    event_id: str
    kind: AudioEventKind
    timestamp: float
    source_type: str
    global_id: int | None = None
    person_id: str = ""
    display_name: str = ""
    camera: str = ""
    zone: str = ""
    direction: str = ""
    dwell_seconds: float | None = None
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_identity(self) -> bool:
        return bool(self.person_id and self.display_name)

    @property
    def subject_key(self) -> str:
        if self.person_id:
            return f"person:{self.person_id}"
        if self.global_id is not None:
            return f"gid:{self.global_id}"
        return f"anonymous:{self.camera or 'unknown'}:{self.kind.value}"

    @property
    def is_generic_zone_event(self) -> bool:
        return self.source_type in _GENERIC_ZONE_TYPES

    def with_identity(self, person_id: str, display_name: str) -> "AudioEvent":
        person_id = _clean_text(person_id, 100)
        display_name = _clean_text(display_name, 80)
        if not person_id or not display_name:
            return self
        return replace(self, person_id=person_id, display_name=display_name)

    @classmethod
    def from_payload(cls, item: Mapping[str, Any]) -> "AudioEvent | None":
        if not isinstance(item, Mapping):
            return None
        event_id = _clean_text(item.get("event_id") or item.get("id"), 200)
        if not event_id:
            return None
        raw_type = _clean_token(item.get("type") or item.get("event") or item.get("kind"))
        zone = _ZONE_ALIASES.get(_clean_token(item.get("zone")), _clean_token(item.get("zone")))
        direction = _clean_token(item.get("direction"))
        kind = canonical_event_kind(raw_type, zone=zone, direction=direction)
        if kind is None:
            return None
        timestamp = _optional_float(item.get("timestamp", item.get("timestamp_s", item.get("created_at"))))
        if timestamp is None or timestamp <= 0:
            return None
        confidence = _optional_float(item.get("confidence"))
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            return None
        dwell_seconds = _optional_float(item.get("dwell_seconds", item.get("dwell_s")))
        if dwell_seconds is not None and dwell_seconds < 0:
            return None
        return cls(
            event_id=event_id,
            kind=kind,
            timestamp=timestamp,
            source_type=raw_type,
            global_id=_optional_int(item.get("global_id", item.get("gid"))),
            person_id=_clean_text(item.get("person_id") or item.get("employee_id"), 100),
            display_name=_clean_text(item.get("display_name") or item.get("name"), 80),
            camera=_clean_text(item.get("camera") or item.get("channel"), 32),
            zone=zone,
            direction=direction,
            dwell_seconds=dwell_seconds,
            confidence=confidence,
        )


def audio_event_stream_present(payload: Mapping[str, Any]) -> bool:
    """An empty list is meaningful: zone mode is active but has no new event."""
    if "audio_events" in payload:
        return isinstance(payload.get("audio_events"), list)
    audio = payload.get("audio")
    return isinstance(audio, Mapping) and isinstance(audio.get("events"), list)


def parse_audio_events(payload: Mapping[str, Any]) -> list[AudioEvent]:
    raw_events: Any
    if "audio_events" in payload:
        raw_events = payload.get("audio_events")
    else:
        audio = payload.get("audio")
        raw_events = audio.get("events") if isinstance(audio, Mapping) else []
    if not isinstance(raw_events, list):
        return []
    return [event for item in raw_events if isinstance(item, Mapping) if (event := AudioEvent.from_payload(item)) is not None]


__all__ = [
    "AudioEvent",
    "AudioEventKind",
    "audio_event_stream_present",
    "canonical_event_kind",
    "parse_audio_events",
]
