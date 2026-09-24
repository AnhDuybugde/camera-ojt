"""Image-space polygon observer that publishes the stable audio event contract.

The tracking process owns geometry.  This module consumes existing confirmed
tracks, uses the bbox bottom-center anchor and never opens a camera stream.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


AUDIO_ZONE_KINDS = frozenset({
    "door_inside", "door_outside", "be_xinh", "water", "restroom",
})
DWELL_SECONDS = {
    "be_xinh": 0.8,
    "water": 1.5,
    "restroom": 1.0,
}


@dataclass(frozen=True, slots=True)
class ImageZone:
    zone_id: str
    label: str
    kind: str
    points: tuple[tuple[float, float], ...]
    color: str = "#22c55e"
    privacy: str = "standard"
    enabled: bool = True

    @property
    def area(self) -> float:
        return abs(sum(
            self.points[index][0] * self.points[(index + 1) % len(self.points)][1]
            - self.points[(index + 1) % len(self.points)][0] * self.points[index][1]
            for index in range(len(self.points))
        )) / 2.0

    def contains(self, point: tuple[float, float]) -> bool:
        polygon = np.asarray(self.points, dtype=np.float32)
        return bool(
            len(polygon) >= 3
            and cv2.pointPolygonTest(polygon, point, False) >= 0
        )


@dataclass(slots=True)
class _SubjectState:
    candidate_zone: str | None = None
    candidate_since_s: float = 0.0
    committed_zone: str | None = None
    committed_since_s: float = 0.0
    emitted_dwell_zone: str | None = None
    last_door_kind: str | None = None
    last_door_at_s: float = float("-inf")
    last_seen_s: float = 0.0


class ZoneEventObserver:
    """Hot-reload polygons and turn stable track transitions into events."""

    def __init__(
        self,
        config_path: Path,
        *,
        transition_dwell_s: float = 0.25,
        door_transition_window_s: float = 8.0,
        event_ttl_s: float = 2.5,
        prune_after_s: float = 15.0,
    ) -> None:
        self.config_path = Path(config_path)
        self.transition_dwell_s = max(0.0, float(transition_dwell_s))
        self.door_transition_window_s = max(1.0, float(door_transition_window_s))
        self.event_ttl_s = max(1.0, float(event_ttl_s))
        self.prune_after_s = max(2.0, float(prune_after_s))
        self.zones: dict[str, tuple[ImageZone, ...]] = {"A": (), "B": ()}
        self._states: dict[tuple[str, int], _SubjectState] = {}
        self._events: deque[tuple[float, dict[str, Any]]] = deque(maxlen=128)
        self._mtime_ns: int | None = None
        self._sequence = 0
        self.last_error = ""
        self.reload(force=True)

    @property
    def enabled(self) -> bool:
        return any(self.zones.values())

    def reload(self, *, force: bool = False) -> bool:
        try:
            mtime_ns = self.config_path.stat().st_mtime_ns
        except OSError as error:
            self.last_error = str(error)
            return False
        if not force and mtime_ns == self._mtime_ns:
            return False
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
            cameras = payload.get("cameras")
            if not isinstance(cameras, Mapping):
                raise ValueError("zone config must contain cameras")
            loaded = {
                channel: self._parse_zones(cameras.get(channel, {}))
                for channel in ("A", "B")
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.last_error = str(error)
            return False
        self.zones = loaded
        self._states.clear()
        self._mtime_ns = mtime_ns
        self.last_error = ""
        return True

    @staticmethod
    def _parse_zones(camera: Any) -> tuple[ImageZone, ...]:
        if not isinstance(camera, Mapping):
            return ()
        result: list[ImageZone] = []
        for raw in camera.get("zones", []):
            if not isinstance(raw, Mapping) or not raw.get("enabled", True):
                continue
            kind = str(raw.get("kind", "custom")).strip().lower()
            points = tuple(
                (float(point[0]), float(point[1]))
                for point in raw.get("points", [])
                if isinstance(point, (list, tuple)) and len(point) == 2
            )
            if len(points) < 3 or any(
                not 0.0 <= coordinate <= 1.0
                for point in points for coordinate in point
            ):
                raise ValueError(f"invalid normalized polygon: {raw.get('id')!r}")
            zone = ImageZone(
                zone_id=str(raw.get("id", "")).strip(),
                label=" ".join(str(raw.get("label", "")).split()),
                kind=kind,
                points=points,
                color=str(raw.get("color", "#22c55e")),
                privacy=str(raw.get("privacy", "standard")),
                enabled=True,
            )
            if not zone.zone_id or zone.area < 0.00005:
                raise ValueError(f"empty/degenerate polygon: {zone.zone_id!r}")
            result.append(zone)
        # Specific/small regions win over broad workspace regions when they overlap.
        return tuple(sorted(result, key=lambda zone: zone.area))

    def observe_tracks(
        self,
        channel: str,
        tracks: Iterable[Any],
        frame_shape: tuple[int, ...],
        *,
        now_s: float,
        wall_time_s: float | None = None,
        aliases: Mapping[int, int] | None = None,
        person_ids: Mapping[int, str] | None = None,
        display_names: Mapping[int, str] | None = None,
    ) -> list[dict[str, Any]]:
        self.reload()
        self._remap_states(aliases or {})
        channel = str(channel).upper()
        wall_time_s = time.time() if wall_time_s is None else float(wall_time_s)
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if width <= 0 or height <= 0:
            return []
        emitted: list[dict[str, Any]] = []
        for track in tracks:
            raw_gid = int(track.track_id)
            gid = int((aliases or {}).get(raw_gid, raw_gid))
            point = (
                max(0.0, min(1.0, float(track.bbox.foot_point[0]) / width)),
                max(0.0, min(1.0, float(track.bbox.foot_point[1]) / height)),
            )
            zone = self.zone_at(channel, point)
            state = self._states.setdefault((channel, gid), _SubjectState())
            state.last_seen_s = now_s
            zone_id = zone.zone_id if zone is not None else None
            if zone_id != state.candidate_zone:
                state.candidate_zone = zone_id
                state.candidate_since_s = now_s
            if zone_id != state.committed_zone and (
                now_s - state.candidate_since_s >= self.transition_dwell_s
            ):
                state.committed_zone = zone_id
                state.committed_since_s = now_s
                state.emitted_dwell_zone = None
                if zone is not None and zone.kind in {"door_inside", "door_outside"}:
                    event = self._door_transition(
                        channel, gid, zone, state, now_s, wall_time_s,
                        person_ids or {}, display_names or {},
                    )
                    if event is not None:
                        emitted.append(event)
            if zone is not None and zone_id == state.committed_zone:
                threshold = DWELL_SECONDS.get(zone.kind)
                dwell = now_s - state.committed_since_s
                if (
                    threshold is not None
                    and dwell >= threshold
                    and state.emitted_dwell_zone != zone.zone_id
                ):
                    event = self._new_event(
                        event_type="ZONE_DWELL",
                        channel=channel,
                        gid=gid,
                        zone=zone,
                        wall_time_s=wall_time_s,
                        now_s=now_s,
                        person_ids=person_ids or {},
                        display_names=display_names or {},
                        dwell_seconds=dwell,
                    )
                    emitted.append(event)
                    state.emitted_dwell_zone = zone.zone_id
        self._prune(now_s)
        return emitted

    def reset(self) -> None:
        self._states.clear()
        self._events.clear()

    def _remap_states(self, aliases: Mapping[int, int]) -> None:
        for (channel, raw_gid), state in list(self._states.items()):
            canonical = int(aliases.get(raw_gid, raw_gid))
            if canonical == raw_gid:
                continue
            old_key, new_key = (channel, raw_gid), (channel, canonical)
            current = self._states.get(new_key)
            if current is None or state.last_seen_s >= current.last_seen_s:
                self._states[new_key] = state
            self._states.pop(old_key, None)

    def zone_at(
        self, channel: str, point: tuple[float, float]
    ) -> ImageZone | None:
        return next(
            (zone for zone in self.zones.get(channel, ()) if zone.contains(point)),
            None,
        )

    def _door_transition(
        self,
        channel: str,
        gid: int,
        zone: ImageZone,
        state: _SubjectState,
        now_s: float,
        wall_time_s: float,
        person_ids: Mapping[int, str],
        display_names: Mapping[int, str],
    ) -> dict[str, Any] | None:
        previous = state.last_door_kind
        previous_at = state.last_door_at_s
        state.last_door_kind = zone.kind
        state.last_door_at_s = now_s
        if previous == zone.kind or now_s - previous_at > self.door_transition_window_s:
            return None
        if previous == "door_outside" and zone.kind == "door_inside":
            event_type, direction = "DOOR_ENTER", "outside_to_inside"
        elif previous == "door_inside" and zone.kind == "door_outside":
            event_type, direction = "DOOR_EXIT", "inside_to_outside"
        else:
            return None
        return self._new_event(
            event_type=event_type,
            channel=channel,
            gid=gid,
            zone=zone,
            wall_time_s=wall_time_s,
            now_s=now_s,
            person_ids=person_ids,
            display_names=display_names,
            direction=direction,
        )

    def _new_event(
        self,
        *,
        event_type: str,
        channel: str,
        gid: int,
        zone: ImageZone,
        wall_time_s: float,
        now_s: float,
        person_ids: Mapping[int, str],
        display_names: Mapping[int, str],
        direction: str = "",
        dwell_seconds: float | None = None,
    ) -> dict[str, Any]:
        self._sequence += 1
        event = {
            "event_id": (
                f"zone-{channel.lower()}-{gid}-{event_type.lower()}-"
                f"{int(wall_time_s * 1000)}-{self._sequence}"
            ),
            "type": event_type,
            "global_id": gid,
            "person_id": str(person_ids.get(gid) or ""),
            "display_name": str(display_names.get(gid) or ""),
            "camera": channel,
            "zone": zone.kind,
            "zone_id": zone.zone_id,
            "timestamp": wall_time_s,
            "confidence": 1.0,
        }
        if direction:
            event["direction"] = direction
        if dwell_seconds is not None:
            event["dwell_seconds"] = round(float(dwell_seconds), 3)
        self._events.append((now_s + self.event_ttl_s, event))
        return event

    def active_events(self, now_s: float) -> list[dict[str, Any]]:
        while self._events and self._events[0][0] < now_s:
            self._events.popleft()
        return [dict(event) for expires, event in self._events if expires >= now_s]

    def occupancy(self) -> list[dict[str, Any]]:
        rows = []
        for (channel, gid), state in sorted(self._states.items()):
            if state.committed_zone is not None:
                rows.append({
                    "camera": channel,
                    "global_id": gid,
                    "zone_id": state.committed_zone,
                })
        return rows

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "config_path": str(self.config_path),
            "last_error": self.last_error,
            "zone_counts": {
                channel: len(zones) for channel, zones in self.zones.items()
            },
            "occupancy": self.occupancy(),
        }

    def draw_overlay(self, frame: np.ndarray, channel: str) -> np.ndarray:
        height, width = frame.shape[:2]
        overlay = frame.copy()
        for zone in self.zones.get(str(channel).upper(), ()):
            polygon = np.asarray([
                (round(x * width), round(y * height)) for x, y in zone.points
            ], dtype=np.int32)
            if len(polygon) < 3:
                continue
            rgb = self._hex_rgb(zone.color)
            bgr = (rgb[2], rgb[1], rgb[0])
            cv2.fillPoly(overlay, [polygon], bgr)
            cv2.polylines(frame, [polygon], True, bgr, 2, cv2.LINE_AA)
            x, y = polygon[0]
            cv2.putText(
                frame, zone.label, (int(x) + 5, max(20, int(y) - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, bgr, 2, cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)
        return frame

    def _prune(self, now_s: float) -> None:
        stale = [
            key for key, state in self._states.items()
            if now_s - state.last_seen_s > self.prune_after_s
        ]
        for key in stale:
            self._states.pop(key, None)
        self.active_events(now_s)

    @staticmethod
    def _hex_rgb(value: str) -> tuple[int, int, int]:
        value = str(value).lstrip("#")
        try:
            return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
        except (TypeError, ValueError):
            return (34, 197, 94)


__all__ = ["AUDIO_ZONE_KINDS", "DWELL_SECONDS", "ImageZone", "ZoneEventObserver"]
