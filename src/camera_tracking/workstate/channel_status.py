"""Business state per channel: exactly 5 states.

IMAGE-ZONE mode (zone_mode set, positions are normalized 0..1 bbox
centers vs R1/R2/R3 polygons in image space):
- channel A ("a_r1"): center in R1 -> AWAY, else WORKING.
- channel B ("b_door"): center in R2 -> OUT_OF_DOOR, elif in R3 ->
  AT_DOOR, else AWAY. R2 wins on overlap.
Raw labels pass through the same grace/dwell StateStabilizer so border
flicker never reaches the database.

ROI mode (workstations configured, --no-image-zones): foot point (floor
meters) vs core/extended ROI -> stabilizer -> WORKING / AWAY. Core and
extended both mean WORKING; there is no near-seat state.

INTERIM mode (no workstations): normalized 0..1 bbox-center displacement
from the first-seen anchor. Sustained displacement past move_ratio flips
to AWAY even while visible; standing still at a new spot adopts it as the
new anchor (-> WORKING).

Without positions at all: pure presence mode (visible in A = WORKING,
visible in B = AWAY).
Absence from view: WORKING/AT_DOOR/OUT_OF_DOOR missing longer than
away_grace_s -> AWAY. AWAY never leaves on its own; only a fresh
position (or prune) changes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot

from camera_tracking.workstate.image_zones import (
    classify_channel_a,
    classify_channel_b,
)
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
    """Business state of one person inside one specific channel. Only 5."""

    UNKNOWN = "UNKNOWN"  # not enough observation yet
    WORKING = "WORKING"  # channel A: bbox center outside R1
    AWAY = "AWAY"  # channel A: center in R1 / channel B default / absent
    AT_DOOR = "AT_DOOR"  # channel B: bbox center in R3
    OUT_OF_DOOR = "OUT_OF_DOOR"  # channel B: bbox center in R2


_OBS_TO_COMMITTED = {
    ZoneObs.AT_CORE: WorkstationState.WORKING,
    ZoneObs.IN_EXTENDED: WorkstationState.WORKING,
    ZoneObs.OUTSIDE: WorkstationState.AWAY,
}


@dataclass
class ChannelBusinessTracker:
    """Business tracker per Global-ID for ONE channel.

    Parameters
    ----------
    channel: channel name ("A" / "B"), display/log only.
    away_grace_s: absent from view longer than N seconds:
        WORKING/AT_DOOR/OUT_OF_DOOR -> AWAY.
    out_after_s: deprecated, ignored (5-state mode has no POSSIBLY_OUT).
    return_stable_s: deprecated, ignored (5-state mode has no RETURNING;
        the stabilizer dwell already debounces the return).
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
    move_ratio: float = 0.0
    settle_ratio: float = 0.02
    # Image-zone mode: "none" | "a_r1" | "b_door". When active, positions
    # MUST be normalized 0..1 bbox centers (same frame as the R polygons).
    zone_mode: str = "none"
    zone_r1: list[tuple[float, float]] = field(default_factory=list)
    zone_r2: list[tuple[float, float]] = field(default_factory=list)
    zone_r3: list[tuple[float, float]] = field(default_factory=list)
    _last_present_s: dict[int, float] = field(default_factory=dict, init=False)
    _last_absent_s: dict[int, float] = field(default_factory=dict, init=False)
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

    @property
    def use_image_zones(self) -> bool:
        if self.zone_mode == "a_r1":
            return len(self.zone_r1) >= 3
        if self.zone_mode == "b_door":
            return len(self.zone_r2) >= 3 and len(self.zone_r3) >= 3
        return False

    def transfer_assignment(self, old_gid: int, canon_gid: int) -> None:
        self._assigner.transfer_assignment(old_gid, canon_gid)
        stab = self._stabilizers.pop(old_gid, None)
        if stab is not None:
            self._stabilizers.setdefault(canon_gid, stab)
        for store in (self._anchors, self._last_pos, self._still_since_s):
            store.pop(old_gid, None)

    def forget(self, gid: int) -> None:
        for store in (self._last_present_s, self._last_absent_s,
                      self._states, self._stabilizers,
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

        positions: image-zone mode -> normalized 0..1 bbox centers;
        workstation mode -> floor-meter foot points (homography).
        persons: optional gid -> person_id for explicit workstation
        overrides.
        """
        positions = positions or {}
        persons = persons or {}
        for gid in present_gids:
            self._last_present_s[gid] = now_s
            self._last_absent_s.pop(gid, None)
            state = self._states.get(gid, PersonBusinessState.UNKNOWN)

            if self.use_image_zones and positions.get(gid) is not None:
                state = self._update_image_zoned(
                    gid, now_s, state, positions[gid])
            elif self.use_zones:
                state = self._update_zoned(
                    gid, now_s, state,
                    positions.get(gid), persons.get(gid))
            elif positions.get(gid) is not None and self.move_ratio > 0:
                state = self._update_displaced(
                    gid, now_s, state, positions[gid])
            elif state is PersonBusinessState.UNKNOWN:
                # Pure presence, no position context: A works, B waits.
                state = (PersonBusinessState.AWAY
                         if (self.zone_mode == "b_door"
                             and self.use_image_zones)
                         else PersonBusinessState.WORKING)
            elif (state is PersonBusinessState.AWAY
                    and not self.use_image_zones):
                # Reappeared with no position context (presence-only /
                # legacy ROI-less mode): visible again = WORKING. With
                # image zones the next zoned position decides instead.
                state = PersonBusinessState.WORKING
            # Otherwise (no fresh position): keep previous state. The
            # stabilizer already debounces the return, so no RETURNING
            # intermediate is needed; absence timers below handle goodbye.
            self._states[gid] = state

        for gid, state in list(self._states.items()):
            if gid in present_gids:
                continue
            absent_since = self._last_absent_s.setdefault(
                gid, self._last_present_s.get(gid, now_s)
            )
            missing = now_s - absent_since
            if (
                state in (PersonBusinessState.WORKING,
                          PersonBusinessState.AT_DOOR,
                          PersonBusinessState.OUT_OF_DOOR)
                and missing >= self.away_grace_s
            ):
                self._states[gid] = PersonBusinessState.AWAY
            # AWAY + still absent: keep. UNKNOWN + absent: keep.
        self._prune(now_s, present_gids)
        return dict(self._states)

    def _update_image_zoned(
        self,
        gid: int,
        now_s: float,
        state: PersonBusinessState,
        center_norm: tuple[float, float],
    ) -> PersonBusinessState:
        """Image-space R1/R2/R3 rule with grace/dwell stabilization.

        center_norm: normalized 0..1 bbox center in the channel frame.
        """
        if self.zone_mode == "a_r1":
            raw_label = classify_channel_a(center_norm, self.zone_r1)
            raw = (WorkstationState.AWAY
                   if raw_label == "AWAY"
                   else WorkstationState.WORKING)
        elif self.zone_mode == "b_door":
            raw_label = classify_channel_b(
                center_norm, self.zone_r2, self.zone_r3)
            if raw_label == "OUT_OF_DOOR":
                raw = WorkstationState.OUT_OF_DOOR
            elif raw_label == "AT_DOOR":
                raw = WorkstationState.AT_DOOR
            else:
                raw = WorkstationState.AWAY
        else:
            return state
        committed = self._stabilizer_for(gid).update(now_s, raw)
        # Stabilizer starts UNKNOWN: hold previous business state until the
        # first commit lands (mirrors workstation-mode behavior).
        if committed is WorkstationState.UNKNOWN:
            return state
        return self._apply_committed(state, committed)

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
        return self._apply_committed(state, committed)

    def _update_displaced(
        self,
        gid: int,
        now_s: float,
        state: PersonBusinessState,
        pos: tuple[float, float],
    ) -> PersonBusinessState:
        """INTERIM mode: large sustained displacement from the anchor flips
        to AWAY even while visible. Standing still at a new spot adopts it
        as the new anchor (-> WORKING)."""
        anchor = self._anchors.get(gid)
        if anchor is None:
            anchor = self._anchors[gid] = pos
        moved = hypot(pos[0] - anchor[0], pos[1] - anchor[1]) > self.move_ratio
        raw = (WorkstationState.AWAY if moved
               else WorkstationState.WORKING)
        committed = self._stabilizer_for(gid).update(now_s, raw)
        if committed is WorkstationState.WORKING and not moved:
            # Settled at anchor: absorb tiny drift (chair shifts, jitter).
            ax, ay = self._anchors[gid]
            self._anchors[gid] = (ax + 0.05 * (pos[0] - ax),
                                  ay + 0.05 * (pos[1] - ay))
        elif committed is not WorkstationState.WORKING:
            # Frame-to-frame stillness far from anchor: sitting at a new
            # spot -> adopt it (next commits flip back to WORKING).
            last = self._last_pos.get(gid)
            if last is None or hypot(pos[0] - last[0],
                                     pos[1] - last[1]) > self.settle_ratio:
                self._still_since_s[gid] = now_s
            else:
                since = self._still_since_s.setdefault(gid, now_s)
                if now_s - since >= self.dwell_s:
                    self._anchors[gid] = pos
                    self._still_since_s.pop(gid, None)
        self._last_pos[gid] = pos
        if committed is WorkstationState.UNKNOWN:
            return state
        return self._apply_committed(state, committed)

    def _stabilizer_for(self, gid: int) -> StateStabilizer:
        stabilizer = self._stabilizers.get(gid)
        if stabilizer is None:
            stabilizer = self._stabilizers[gid] = StateStabilizer(
                grace_s=self.grace_s, dwell_s=self.dwell_s,
                motion_influence=self.motion_influence)
        return stabilizer

    @staticmethod
    def _apply_committed(
        state: PersonBusinessState,
        committed: WorkstationState,
    ) -> PersonBusinessState:
        if committed is WorkstationState.WORKING:
            if state is PersonBusinessState.UNKNOWN:
                return PersonBusinessState.WORKING
            if state in (PersonBusinessState.AWAY,
                         PersonBusinessState.AT_DOOR,
                         PersonBusinessState.OUT_OF_DOOR):
                return PersonBusinessState.WORKING
            return state
        if committed is WorkstationState.AWAY:
            return PersonBusinessState.AWAY
        if committed is WorkstationState.AT_DOOR:
            return PersonBusinessState.AT_DOOR
        if committed is WorkstationState.OUT_OF_DOOR:
            return PersonBusinessState.OUT_OF_DOOR
        return state

    def state_of(self, global_id: int) -> PersonBusinessState:
        return self._states.get(global_id, PersonBusinessState.UNKNOWN)


__all__ = ["ChannelBusinessTracker", "PersonBusinessState"]
