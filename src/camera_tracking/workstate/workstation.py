"""Workstation ROI assignment + raw zone observation (floor meters).

Position source: bbox bottom-center (foot point) mapped through the room
homography into floor meters. Motion, direction, pose or distance are NEVER
conditions here; they may only modulate stabilizer timing elsewhere.

Per person, one workstation has:
- core ROI: desk/chair area -> WORKING
- extended ROI: nearby area -> WORKING (collapsed, no near-seat state)
- outside extended -> AWAY

Assignment: whoever dwells in a core longer than assign_dwell_s owns it;
an explicit person_id -> workstation map always wins (stable across
Global IDs) and survives identity merges via transfer_assignment().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np


class ZoneObs(str, Enum):
    AT_CORE = "AT_CORE"
    IN_EXTENDED = "IN_EXTENDED"
    OUTSIDE = "OUTSIDE"
    UNKNOWN = "UNKNOWN"  # no position (occluded) or no workstation context


@dataclass(frozen=True, slots=True)
class WorkstationZone:
    name: str
    core: list[tuple[float, float]]
    extended: list[tuple[float, float]]


def _signed_distance(point: tuple[float, float],
                     polygon: list[tuple[float, float]]) -> float:
    """Signed distance in meters (>0 inside). Empty polygon -> -inf."""
    if len(polygon) < 3:
        return float("-inf")
    contour = np.asarray(polygon, dtype=np.float32)
    return float(cv2.pointPolygonTest(
        contour, (float(point[0]), float(point[1])), True))


def observe_zone(
    point: tuple[float, float] | None,
    zone: WorkstationZone,
    hysteresis_m: float = 0.0,
    currently_inside: bool = False,
) -> ZoneObs:
    """Raw zone of one floor point against one workstation.

    Spatial hysteresis: while inside, the point must leave by more than
    hysteresis_m before it counts as out (border flicker immunity).
    """
    if point is None:
        return ZoneObs.UNKNOWN
    margin = hysteresis_m if currently_inside else 0.0
    if _signed_distance(point, zone.core) >= -margin:
        return ZoneObs.AT_CORE
    if _signed_distance(point, zone.extended) >= -margin:
        return ZoneObs.IN_EXTENDED
    return ZoneObs.OUTSIDE


@dataclass
class WorkstationAssigner:
    """Bind each Global ID to at most one workstation."""

    zones: list[WorkstationZone] = field(default_factory=list)
    assign_dwell_s: float = 5.0
    hysteresis_m: float = 0.3
    person_map: dict[str, str] = field(default_factory=dict)
    assigned: dict[int, str] = field(default_factory=dict, init=False)
    _dwell_since: dict[tuple[int, str], float] = field(
        default_factory=dict, init=False)

    def zone_of(self, name: str) -> WorkstationZone | None:
        for zone in self.zones:
            if zone.name == name:
                return zone
        return None

    def transfer_assignment(self, old_gid: int, canon_gid: int) -> None:
        """Identity merge: the canonical gid inherits the workstation."""
        if old_gid == canon_gid:
            return
        name = self.assigned.pop(old_gid, None)
        if name is not None:
            self.assigned.setdefault(canon_gid, name)
        for key in [k for k in self._dwell_since if k[0] == old_gid]:
            self._dwell_since.pop(key, None)

    def forget(self, gid: int) -> None:
        self.assigned.pop(gid, None)
        for key in [k for k in self._dwell_since if k[0] == gid]:
            self._dwell_since.pop(key, None)

    def observe(
        self,
        gid: int,
        point: tuple[float, float] | None,
        now_s: float,
        person_id: str | None = None,
    ) -> tuple[WorkstationZone | None, ZoneObs]:
        """Return (workstation in context, raw observation).

        Explicit person_map wins immediately. Otherwise the currently
        assigned workstation is tested first (stickiness); an unassigned
        gid dwelling in some core long enough adopts it.
        """
        if person_id is not None and person_id in self.person_map:
            zone = self.zone_of(self.person_map[person_id])
            if zone is not None:
                self.assigned[gid] = zone.name
                return zone, observe_zone(
                    point, zone, self.hysteresis_m,
                    currently_inside=True)
            return None, ZoneObs.UNKNOWN

        assigned_name = self.assigned.get(gid)
        if assigned_name is not None:
            zone = self.zone_of(assigned_name)
            if zone is None:
                self.assigned.pop(gid, None)
            else:
                return zone, observe_zone(
                    point, zone, self.hysteresis_m,
                    currently_inside=True)

        # Unassigned: test every core for a new home.
        if point is not None:
            for zone in self.zones:
                if _signed_distance(point, zone.core) >= 0:
                    key = (gid, zone.name)
                    since = self._dwell_since.setdefault(key, now_s)
                    if now_s - since >= self.assign_dwell_s:
                        self.assigned[gid] = zone.name
                        self._dwell_since.pop(key, None)
                        return zone, ZoneObs.AT_CORE
                    return zone, ZoneObs.AT_CORE
            for key in [k for k in self._dwell_since if k[0] == gid]:
                self._dwell_since.pop(key, None)
            for zone in self.zones:
                if _signed_distance(point, zone.extended) >= 0:
                    return zone, ZoneObs.IN_EXTENDED
        return None, ZoneObs.UNKNOWN if point is None else ZoneObs.OUTSIDE


__all__ = [
    "WorkstationAssigner",
    "WorkstationZone",
    "ZoneObs",
    "observe_zone",
]
