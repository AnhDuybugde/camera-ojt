"""Business state per channel: workstation ROI when calibrated, else
interim displacement mode (no floor mapping yet).

ROI mode (workstations configured): foot point (floor meters) vs core /
extended ROI -> stabilizer -> AT / NEAR / AWAY. Motion is never a
condition there.

INTERIM mode (no workstations): normalized 0..1 bbox-center displacement
from the first-seen anchor. Sustained displacement past move_ratio flips
to AWAY even while visible; sitting still at a new spot adopts it as the
new anchor (-> RETURNING -> WORKING). This is intentionally coarse until
the camera is fixed and ROIs are drawn; thresholds live in workstate
config (move_ratio, settle_ratio).

Without positions at all: pure presence mode (visible = WORKING).
Absence from view always runs the AWAY_TEMP -> POSSIBLY_OUT timers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot

from camera_tracking.workstate.stabilizer import (
    StateStabilizer,
    WorkstationState,
)
from camera_tracking.workstate.workstation import (
    WorkstationAssigner,
    WorkstationZone,
    ZoneObs,
)


class PersonBusinessState(str, Enum):
    """Business state of one person inside one specific channel."""

    UNKNOWN = "UNKNOWN"  # not enough observation yet
    WORKING = "WORKING"  # committed AT_WORKSTATION (or visible, no ROIs)
    NEAR_SEAT = "NEAR_SEAT"  # committed NEAR_WORKSTATION
    AWAY_TEMP = "AWAY_TEMP"  # away from seat (zone or absence grace passed)
    POSSIBLY_OUT = "POSSIBLY_OUT"  # absent long, possibly left
    RETURNING = "RETURNING"  # back at seat, stabilizing


_OBS_TO_COMMITTED = {
    ZoneObs.AT_CORE: WorkstationState.AT_WORKSTATION,
    ZoneObs.IN_EXTENDED: WorkstationState.NEAR_WORKSTATION,
    ZoneObs.OUTSIDE: WorkstationState.AWAY_FROM_WORKSTATION,
}


@dataclass
class ChannelBusinessTracker:
    """Business tracker per Global-ID for ONE channel.

    Parameters
    ----------
    channel: channel name ("A" / "B"), display/log only.
    away_grace_s: absent from view longer than N seconds:
        WORKING/NEAR/RETURNING -> AWAY_TEMP.
    out_after_s: absent longer than N seconds:
        AWAY_TEMP -> POSSIBLY_OUT.
    return_stable_s: back at seat stable for N seconds:
        RETURNING -> WORKING.
    workstations: workstation zones (floor meters). Empty = presence-only.
    grace_s / dwell_s / hysteresis_m / motion_influence: stabilizer tuning.
    assign_dwell_s: dwell in a core before adopting it as workstation.
    person_map: person_id -> workstation name override.
    prune_after_s: forget per-ID memory after this absence (0 disables).
    """

    channel: str = "A"
    away_grace_s: float = 1.5
    out_after_s: float = 20.0
    return_stable_s: float = 2.0
    workstations: list[WorkstationZone] = field(default_factory=list)
    grace_s: float = 1.5
    dwell_s: float = 2.0
    assign_dwell_s: float = 5.0
    hysteresis_m: float = 0.3
    motion_influence: float = 0.0
    person_map: dict[str, str] = field(default_factory=dict)
    prune_after_s: float = 300.0
    # Interim displacement mode (positions in normalized 0..1 units).
    move_ratio: float = 0.15
    settle_ratio: float = 0.02
    _last_present_s: dict[int, float] = field(default_factory=dict, init=False)
    _last_absent_s: dict[int, float] = field(default_factory=dict, init=False)
    _returned_at_s: dict[int, float] = field(default_factory=dict, init=False)
    _states: dict[int, PersonBusinessState] = field(default_factory=dict, init=False)
    _stabilizers: dict[int, StateStabilizer] = field(default_factory=dict, init=False)
    _anchors: dict[int, tuple[float, float]] = field(default_factory=dict, init=False)
    _last_pos: dict[int, tuple[float, float]] = field(default_factory=dict, init=False)
    _still_since_s: dict[int, float] = field(default_factory=dict, init=False)
    _assigner: WorkstationAssigner = field(default_factory=WorkstationAssigner,
                                          init=False)

    def __post_init__(self) -> None:
        self._assigner = WorkstationAssigner(
            zones=list(self.workstations),
            assign_dwell_s=self.assign_dwell_s,
            hysteresis_m=self.hysteresis_m,
            person_map=dict(self.person_map),
        )

    @property
    def use_zones(self) -> bool:
        return len(self._assigner.zones) > 0

    def transfer_assignment(self, old_gid: int, canon_gid: int) -> None:
        self._assigner.transfer_assignment(old_gid, canon_gid)
        stab = self._stabilizers.pop(old_gid, None)
        if stab is not None:
            self._stabilizers.setdefault(canon_gid, stab)
        for store in (self._anchors, self._last_pos, self._still_since_s):
            store.pop(old_gid, None)

    def forget(self, gid: int) -> None:
        for store in (self._last_present_s, self._last_absent_s,
                      self._returned_at_s, self._states, self._stabilizers,
                      self._anchors, self._last_pos, self._still_since_s):
            store.pop(gid, None)
        self._assigner.forget(gid)

    def forget_retired(self, alive_gids: set[int]) -> int:
        """Drop memory for IDs the identity manager already retired."""
        dropped = 0
        for gid in [g for g in list(self._states) if g not in alive_gids]:
            self.forget(gid)
            dropped += 1
        return dropped

    def _prune(self, now_s: float, present_gids: set[int]) -> None:
        if self.prune_after_s <= 0:
            return
        for gid in list(self._states):
            if gid in present_gids:
                continue
            if now_s - self._last_present_s.get(gid, now_s) >= self.prune_after_s:
                self.forget(gid)

    def update(
        self,
        now_s: float,
        present_gids: set[int],
        positions: dict[int, tuple[float, float]] | None = None,
        persons: dict[int, str] | None = None,
    ) -> dict[int, PersonBusinessState]:
        """Update from Global IDs seen in this channel.

        positions: floor-meter foot points (homography). persons: optional
        gid -> person_id for explicit workstation overrides.
        """
        positions = positions or {}
        persons = persons or {}
        for gid in present_gids:
            self._last_present_s[gid] = now_s
            self._last_absent_s.pop(gid, None)
            state = self._states.get(gid, PersonBusinessState.UNKNOWN)

            if self.use_zones:
                state = self._update_zoned(
                    gid, now_s, state,
                    positions.get(gid), persons.get(gid))
            elif positions.get(gid) is not None and self.move_ratio > 0:
                state = self._update_displaced(
                    gid, now_s, state, positions[gid])
            elif state is PersonBusinessState.UNKNOWN:
                state = PersonBusinessState.WORKING
            elif state in (PersonBusinessState.AWAY_TEMP,
                           PersonBusinessState.POSSIBLY_OUT):
                # Reappeared after absence (no position context).
                state = PersonBusinessState.RETURNING
                self._returned_at_s[gid] = now_s
            self._states[gid] = state

            if state is PersonBusinessState.RETURNING and (
                    now_s - self._returned_at_s.get(gid, now_s)
                    >= self.return_stable_s):
                self._states[gid] = PersonBusinessState.WORKING

        for gid, state in list(self._states.items()):
            if gid in present_gids:
                continue
            absent_since = self._last_absent_s.setdefault(
                gid, self._last_present_s.get(gid, now_s)
            )
            missing = now_s - absent_since
            if (
                state in (PersonBusinessState.WORKING,
                          PersonBusinessState.NEAR_SEAT,
                          PersonBusinessState.RETURNING)
                and missing >= self.away_grace_s
            ):
                self._states[gid] = PersonBusinessState.AWAY_TEMP
            elif (
                state is PersonBusinessState.AWAY_TEMP
                and missing >= self.out_after_s
            ):
                self._states[gid] = PersonBusinessState.POSSIBLY_OUT
            # POSSIBLY_OUT + still absent: keep, wait for return.
        self._prune(now_s, present_gids)
        return dict(self._states)

    def _update_zoned(
        self,
        gid: int,
        now_s: float,
        state: PersonBusinessState,
        point: tuple[float, float] | None,
        person_id: str | None,
    ) -> PersonBusinessState:
        """Position-driven transition. No motion term anywhere."""
        _zone, obs = self._assigner.observe(gid, point, now_s, person_id)
        if obs is ZoneObs.UNKNOWN:
            return state  # no position: never decide, keep previous state
        stabilizer = self._stabilizer_for(gid)
        committed = stabilizer.update(now_s, _OBS_TO_COMMITTED[obs])
        return self._apply_committed(gid, now_s, state, committed)

    def _update_displaced(
        self,
        gid: int,
        now_s: float,
        state: PersonBusinessState,
        pos: tuple[float, float],
    ) -> PersonBusinessState:
        """INTERIM mode: large sustained displacement from the anchor flips
        to AWAY even while visible. Standing still at a new spot adopts it
        as the new anchor (-> RETURNING -> WORKING)."""
        anchor = self._anchors.get(gid)
        if anchor is None:
            anchor = self._anchors[gid] = pos
        moved = hypot(pos[0] - anchor[0], pos[1] - anchor[1]) > self.move_ratio
        raw = (WorkstationState.AWAY_FROM_WORKSTATION if moved
               else WorkstationState.AT_WORKSTATION)
        committed = self._stabilizer_for(gid).update(now_s, raw)
        if committed is WorkstationState.AT_WORKSTATION and not moved:
            # Settled at anchor: absorb tiny drift (chair shifts, jitter).
            ax, ay = self._anchors[gid]
            self._anchors[gid] = (ax + 0.05 * (pos[0] - ax),
                                  ay + 0.05 * (pos[1] - ay))
        elif committed is not WorkstationState.AT_WORKSTATION:
            # Frame-to-frame stillness far from anchor: sitting at a new
            # spot -> adopt it (next frames flip RETURNING -> WORKING).
            last = self._last_pos.get(gid)
            if last is None or hypot(pos[0] - last[0],
                                     pos[1] - last[1]) > self.settle_ratio:
                self._still_since_s[gid] = now_s
            else:
                since = self._still_since_s.setdefault(gid, now_s)
                if now_s - since >= self.return_stable_s:
                    self._anchors[gid] = pos
                    self._still_since_s.pop(gid, None)
        self._last_pos[gid] = pos
        return self._apply_committed(gid, now_s, state, committed)

    def _stabilizer_for(self, gid: int) -> StateStabilizer:
        stabilizer = self._stabilizers.get(gid)
        if stabilizer is None:
            stabilizer = self._stabilizers[gid] = StateStabilizer(
                grace_s=self.grace_s, dwell_s=self.dwell_s,
                motion_influence=self.motion_influence)
        return stabilizer

    def _apply_committed(
        self,
        gid: int,
        now_s: float,
        state: PersonBusinessState,
        committed: WorkstationState,
    ) -> PersonBusinessState:
        if committed is WorkstationState.AT_WORKSTATION:
            if state in (PersonBusinessState.AWAY_TEMP,
                         PersonBusinessState.POSSIBLY_OUT,
                         PersonBusinessState.NEAR_SEAT):
                self._returned_at_s[gid] = now_s
                return PersonBusinessState.RETURNING
            return state if state is not PersonBusinessState.UNKNOWN \
                else PersonBusinessState.WORKING
        if committed is WorkstationState.NEAR_WORKSTATION:
            if state in (PersonBusinessState.WORKING,
                         PersonBusinessState.RETURNING,
                         PersonBusinessState.UNKNOWN,
                         PersonBusinessState.POSSIBLY_OUT):
                return PersonBusinessState.NEAR_SEAT
            return state
        if committed is WorkstationState.AWAY_FROM_WORKSTATION:
            if state in (PersonBusinessState.WORKING,
                         PersonBusinessState.NEAR_SEAT,
                         PersonBusinessState.RETURNING):
                return PersonBusinessState.AWAY_TEMP
            return state
        return state

    def state_of(self, global_id: int) -> PersonBusinessState:
        return self._states.get(global_id, PersonBusinessState.UNKNOWN)


__all__ = ["ChannelBusinessTracker", "PersonBusinessState"]
