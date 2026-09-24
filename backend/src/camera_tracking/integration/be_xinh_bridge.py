"""Bridge the tracking status API to the event-driven Be Xinh companion.

The bridge deliberately stays outside ``run_workstate.py``.  The model team can
keep changing YOLO, ByteTrack, Re-ID and face recognition without creating a
large merge conflict with the audio implementation.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.request import Request, urlopen

from camera_tracking.audio.events import (
    AudioEvent,
    AudioEventKind,
    audio_event_stream_present,
    parse_audio_events,
)


@dataclass(frozen=True, slots=True)
class PersonSnapshot:
    """Small, validated projection of one person in ``/status.json``."""

    global_id: int
    person_id: str
    display_name: str
    in_room: bool
    label: str
    stationary_for_s: float | None
    near_camera: bool | None
    estimated_distance_m: float | None


@dataclass(frozen=True, slots=True)
class ActivityEvent:
    """A short-lived gesture or motion event emitted by the model process."""

    sequence: int
    kind: str
    global_id: int
    channel: str
    confidence: float


class Companion(Protocol):
    """Subset of :class:`HamyCompanion` required by this adapter."""

    def identify(
        self,
        *,
        global_id: int,
        person_id: str,
        display_name: str,
        now_s: float,
    ) -> None: ...

    def observe_presence(
        self,
        *,
        person_id: str,
        display_name: str,
        now_s: float,
        stationary_since_s: float | None = None,
        global_id: int | None = None,
        near_camera: bool | None = None,
        estimated_distance_m: float | None = None,
    ) -> None: ...

    def tick(self, now_s: float) -> None: ...

    def forget_global_ids(self, alive_global_ids: set[int]) -> None: ...

    def wave(
        self,
        *,
        now_s: float,
        person_id: str | None = None,
        display_name: str | None = None,
        global_id: int | None = None,
    ) -> bool: ...

    def motion_event(
        self,
        *,
        kind: str,
        now_s: float,
        global_id: int,
        person_id: str | None,
        display_name: str | None,
    ) -> bool: ...

    def set_auto_arrival_from_presence(self, enabled: bool) -> None: ...


class EventRouter(Protocol):
    @property
    def enabled(self) -> bool: ...

    def route_many(
        self,
        events: list[AudioEvent],
        *,
        now_s: float,
        wall_time_s: float,
        resolve_identity,
    ) -> list: ...


def parse_people(payload: dict[str, Any]) -> list[PersonSnapshot]:
    """Return only identified people with a usable Global ID.

    Unknown detections are intentionally ignored: speaking a guessed name is
    worse than remaining silent.
    """

    raw_people = payload.get("people", [])
    if not isinstance(raw_people, list):
        return []

    people: list[PersonSnapshot] = []
    for item in raw_people:
        if not isinstance(item, dict):
            continue
        tracking_state = str(item.get("tracking_state") or "ACTIVE").upper()
        if tracking_state != "ACTIVE":
            continue
        try:
            global_id = int(item["gid"])
        except (KeyError, TypeError, ValueError):
            continue
        person_id = str(item.get("person_id") or "").strip()
        display_name = str(item.get("name") or "").strip()
        if not person_id or not display_name:
            continue
        stationary_for_s: float | None = None
        try:
            raw_stationary = item.get("stationary_for_s")
            if raw_stationary is not None:
                stationary_for_s = max(0.0, float(raw_stationary))
        except (TypeError, ValueError):
            stationary_for_s = None
        raw_near = item.get("near_camera")
        near_camera = raw_near if isinstance(raw_near, bool) else None
        estimated_distance_m: float | None = None
        try:
            raw_distance = item.get("estimated_distance_m")
            if raw_distance is not None:
                estimated_distance_m = max(0.0, float(raw_distance))
        except (TypeError, ValueError):
            estimated_distance_m = None
        people.append(
            PersonSnapshot(
                global_id=global_id,
                person_id=person_id,
                display_name=display_name,
                in_room=bool(item.get("in_room", True)),
                label=str(item.get("label") or ""),
                stationary_for_s=stationary_for_s,
                near_camera=near_camera,
                estimated_distance_m=estimated_distance_m,
            )
        )
    return people


def parse_activity_events(
    payload: dict[str, Any],
    field: str,
    allowed_kinds: set[str],
) -> list[ActivityEvent]:
    raw_events = payload.get(field, [])
    if not isinstance(raw_events, list):
        return []
    events: list[ActivityEvent] = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        try:
            sequence = int(item["seq"])
            global_id = int(item["gid"])
            confidence = min(1.0, max(0.0, float(item.get("confidence", 1.0))))
        except (KeyError, TypeError, ValueError):
            continue
        kind = str(item.get("kind") or "").strip().upper()
        if sequence < 0 or global_id < 0 or kind not in allowed_kinds:
            continue
        events.append(
            ActivityEvent(
                sequence=sequence,
                kind=kind,
                global_id=global_id,
                channel=str(item.get("channel") or "").strip().upper(),
                confidence=confidence,
            )
        )
    return events


class BackendStatusClient:
    """Read the backend status endpoint using only the Python standard library."""

    def __init__(self, status_url: str, timeout_s: float = 2.0) -> None:
        self.status_url = status_url.strip()
        self.timeout_s = max(0.1, float(timeout_s))

    def fetch(self) -> dict[str, Any]:
        request = Request(
            self.status_url,
            headers={"Accept": "application/json", "User-Agent": "be-xinh-bridge/1"},
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            if getattr(response, "status", 200) != 200:
                raise RuntimeError(f"tracking backend returned HTTP {response.status}")
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise TypeError("tracking backend status must be a JSON object")
        return payload


class BeXinhStatusBridge:
    """Translate status snapshots into legacy or Zone Mode audio actions."""

    def __init__(
        self,
        companion: Companion,
        event_router: EventRouter | None = None,
        *,
        identity_cache_ttl_s: float = 300.0,
    ) -> None:
        self.companion = companion
        self.event_router = event_router
        self.identity_cache_ttl_s = max(30.0, float(identity_cache_ttl_s))
        self._bindings: dict[int, tuple[str, str]] = {}
        self._identity_cache: dict[int, tuple[str, str, float]] = {}
        self._seen_events: set[tuple[str, int]] = set()
        self._event_order: deque[tuple[str, int]] = deque(maxlen=256)
        self._presence_arrival_default = bool(
            getattr(companion, "auto_arrival_from_presence", True)
        )
        self.zone_event_mode = False
        self.last_event_results: list = []

    def process(
        self,
        payload: dict[str, Any],
        now_s: float | None = None,
        wall_time_s: float | None = None,
    ) -> int:
        now_s = time.monotonic() if now_s is None else float(now_s)
        wall_time_s = time.time() if wall_time_s is None else float(wall_time_s)
        alive_global_ids: set[int] = set()

        event_stream = audio_event_stream_present(payload)
        router_enabled = bool(
            self.event_router is not None
            and getattr(self.event_router, "enabled", True)
        )
        self.zone_event_mode = bool(event_stream and router_enabled)
        setter = getattr(self.companion, "set_auto_arrival_from_presence", None)
        if callable(setter):
            setter(False if self.zone_event_mode else self._presence_arrival_default)

        people = parse_people(payload)
        for person in people:
            binding = (person.person_id, person.display_name)
            self._identity_cache[person.global_id] = (*binding, now_s)
            if not person.in_room:
                continue
            alive_global_ids.add(person.global_id)
            if self._bindings.get(person.global_id) != binding:
                self.companion.identify(
                    global_id=person.global_id,
                    person_id=person.person_id,
                    display_name=person.display_name,
                    now_s=now_s,
                )
                self._bindings[person.global_id] = binding

            self.companion.observe_presence(
                person_id=person.person_id,
                display_name=person.display_name,
                now_s=now_s,
                stationary_since_s=(
                    now_s - person.stationary_for_s
                    if person.stationary_for_s is not None
                    else None
                ),
                global_id=person.global_id,
                near_camera=person.near_camera,
                estimated_distance_m=person.estimated_distance_m,
            )

        if self.zone_event_mode and self.event_router is not None:
            routed_events = parse_audio_events(payload)
            explicit_wave_gids = {
                event.global_id
                for event in routed_events
                if event.kind is AudioEventKind.WAVE and event.global_id is not None
            }
            # Transition support: a model may publish the new audio_events key
            # before it emits semantic WAVE. Convert the existing OPEN_PALM
            # stream, but suppress it when the same GID already has a WAVE.
            for legacy in parse_activity_events(
                payload, "gesture_events", {"OPEN_PALM"}
            ):
                if not self._accept_event("gesture", legacy.sequence):
                    continue
                if legacy.global_id in explicit_wave_gids:
                    continue
                routed_events.append(AudioEvent(
                    event_id=f"legacy-gesture-{legacy.sequence}",
                    kind=AudioEventKind.WAVE,
                    timestamp=wall_time_s,
                    source_type="wave",
                    global_id=legacy.global_id,
                    camera=legacy.channel,
                    confidence=legacy.confidence,
                ))
            self.last_event_results = self.event_router.route_many(
                routed_events,
                now_s=now_s,
                wall_time_s=wall_time_s,
                resolve_identity=self._resolve_identity,
            )
        else:
            self.last_event_results = []
            for event in parse_activity_events(
                payload, "gesture_events", {"OPEN_PALM"}
            ):
                if not self._accept_event("gesture", event.sequence):
                    continue
                person_id, display_name = self._bindings.get(
                    event.global_id, (None, None)
                )
                self.companion.wave(
                    now_s=now_s,
                    person_id=person_id,
                    display_name=display_name,
                    global_id=event.global_id,
                )

            for event in parse_activity_events(
                payload, "motion_events", {"APPROACH", "STAND_UP"}
            ):
                if not self._accept_event("motion", event.sequence):
                    continue
                person_id, display_name = self._bindings.get(
                    event.global_id, (None, None)
                )
                self.companion.motion_event(
                    kind=event.kind,
                    now_s=now_s,
                    global_id=event.global_id,
                    person_id=person_id,
                    display_name=display_name,
                )

        self.companion.forget_global_ids(alive_global_ids)
        self.companion.tick(now_s)

        retired = [gid for gid in self._bindings if gid not in alive_global_ids]
        for gid in retired:
            self._bindings.pop(gid, None)
        self._purge_identity_cache(now_s)
        return len(alive_global_ids)

    def _resolve_identity(self, global_id: int) -> tuple[str, str] | None:
        cached = self._identity_cache.get(int(global_id))
        if cached is None:
            return None
        person_id, display_name, _last_seen_s = cached
        return person_id, display_name

    def _purge_identity_cache(self, now_s: float) -> None:
        cutoff = now_s - self.identity_cache_ttl_s
        for gid in [
            gid
            for gid, (_person_id, _name, last_seen_s) in self._identity_cache.items()
            if last_seen_s < cutoff
        ]:
            self._identity_cache.pop(gid, None)

    def _accept_event(self, category: str, sequence: int) -> bool:
        key = (category, int(sequence))
        if key in self._seen_events:
            return False
        if len(self._event_order) == self._event_order.maxlen:
            oldest = self._event_order.popleft()
            self._seen_events.discard(oldest)
        self._event_order.append(key)
        self._seen_events.add(key)
        return True
