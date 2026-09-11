from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class StablePersonCount:
    """Apply asymmetric hysteresis so brief misses do not flicker the count."""

    rise_frames: int = 2
    fall_frames: int = 6
    value: int = 0
    _candidate: int | None = field(default=None, init=False)
    _candidate_frames: int = field(default=0, init=False)

    def update(self, observed: int) -> int:
        observed = max(0, observed)
        if observed == self.value:
            self._candidate = None
            self._candidate_frames = 0
            return self.value

        if observed != self._candidate:
            self._candidate = observed
            self._candidate_frames = 1
        else:
            self._candidate_frames += 1

        required_frames = self.rise_frames if observed > self.value else self.fall_frames
        if self._candidate_frames >= max(1, required_frames):
            self.value = observed
            self._candidate = None
            self._candidate_frames = 0
        return self.value
