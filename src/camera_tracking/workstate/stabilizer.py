"""State Stabilizer: raw observation -> stable -> committed.

Three visibly separated layers so detector/camera misses never make the
committed state flap:
- raw: zone observation of this frame (may flicker at borders).
- stable: follows raw only after it persists past grace_s
  (leaving a zone for less than grace keeps the old stable state).
- committed: follows stable only after it persists past dwell_s
  (minimum dwell time). ONLY committed states reach the database.

Motion/speed/pose are never conditions: an optional urgency 0..1 may
only shrink grace/dwell down to 50% (motion_influence scales it);
urgency 0 (default) means pure position + time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class WorkstationState(str, Enum):
    AT_WORKSTATION = "AT_WORKSTATION"
    NEAR_WORKSTATION = "NEAR_WORKSTATION"
    AWAY_FROM_WORKSTATION = "AWAY_FROM_WORKSTATION"
    UNKNOWN = "UNKNOWN"


@dataclass
class StateStabilizer:
    grace_s: float = 3.0
    dwell_s: float = 2.0
    motion_influence: float = 0.0
    state: WorkstationState = field(
        default=WorkstationState.UNKNOWN, init=False)
    _stable: WorkstationState = field(
        default=WorkstationState.UNKNOWN, init=False)
    _raw_since: float | None = field(default=None, init=False)
    _candidate: WorkstationState | None = field(default=None, init=False)
    _candidate_since: float | None = field(default=None, init=False)

    def _scale(self, urgency: float) -> float:
        urgency = min(1.0, max(0.0, urgency)) * self.motion_influence
        return 1.0 - 0.5 * urgency  # urgency only shortens, never extends

    def update(
        self,
        now_s: float,
        raw: WorkstationState,
        urgency: float = 0.0,
    ) -> WorkstationState:
        """Feed one raw observation, return the committed state."""
        scale = self._scale(urgency)
        # Layer 1+2: raw -> stable (grace against flicker-out).
        if raw != self._stable:
            if self._raw_since is None:
                self._raw_since = now_s
            elif now_s - self._raw_since >= self.grace_s * scale:
                self._stable = raw
                self._raw_since = None
        else:
            self._raw_since = None
        # Layer 3: stable -> committed (minimum dwell time).
        if self._stable != self.state:
            if self._candidate != self._stable:
                self._candidate = self._stable
                self._candidate_since = now_s
            elif (self._candidate_since is not None
                    and now_s - self._candidate_since >= self.dwell_s * scale):
                self.state = self._stable
                self._candidate = None
                self._candidate_since = None
        else:
            self._candidate = None
            self._candidate_since = None
        return self.state


__all__ = ["StateStabilizer", "WorkstationState"]
