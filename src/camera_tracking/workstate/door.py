"""Directional door transitions; disappearance never implies an exit."""
from dataclasses import dataclass


@dataclass
class Passage:
    candidate: str = ""
    candidate_since: float = 0
    stable: str = ""
    origin: str = ""
    last_seen: float = 0


class DoorTransitions:
    def __init__(self, dwell_s=0.3, max_gap_s=2.0):
        self.dwell_s, self.max_gap_s = dwell_s, max_gap_s
        self.tracks = {}

    def update(self, key, zone, now_s, *, uncertain=False):
        self.expire(now_s)
        if uncertain or zone not in {"inside", "door", "outside"}:
            self.tracks.pop(key, None)
            return None
        state = self.tracks.setdefault(key, Passage(last_seen=now_s))
        state.last_seen = now_s
        if state.candidate != zone:
            state.candidate, state.candidate_since = zone, now_s
        if now_s - state.candidate_since < self.dwell_s or zone == state.stable:
            return None
        previous, state.stable = state.stable, zone
        if zone == "door":
            state.origin = previous if previous in {"inside", "outside"} else ""
            return None
        origin, state.origin = state.origin, ""
        if previous == "door" and origin and origin != zone:
            return "CHECK_IN" if zone == "inside" else "CHECK_OUT"
        return None

    def expire(self, now_s):
        for key in list(self.tracks):
            if now_s - self.tracks[key].last_seen > self.max_gap_s:
                del self.tracks[key]
