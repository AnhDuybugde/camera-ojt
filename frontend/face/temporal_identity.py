"""Temporal voting and short-lived identity cache for live face tracks."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from face.recognizer import Match


@dataclass(frozen=True, slots=True)
class IdentityDecision:
    match: Match
    status: str  # confirmed | cached | verifying | unknown


@dataclass(slots=True)
class _TrackVotes:
    votes: deque[tuple[float, Match]] = field(default_factory=deque)
    confirmed: Match | None = None
    expires_at: float = 0.0


class TemporalIdentityResolver:
    def __init__(
        self, *, required_votes: int = 2, confirm_ratio: float = 0.67,
        window_seconds: float = 12.0, ttl_seconds: float = 15.0,
    ) -> None:
        self.required_votes = max(1, required_votes)
        self.confirm_ratio = min(1.0, max(0.0, confirm_ratio))
        self.window_seconds = max(1.0, window_seconds)
        self.ttl_seconds = max(1.0, ttl_seconds)
        self._tracks: dict[int, _TrackVotes] = {}

    def update(self, track_id: int, candidate: Match, now: float) -> IdentityDecision:
        state = self._tracks.setdefault(track_id, _TrackVotes())
        state.votes.append((now, candidate))
        cutoff = now - self.window_seconds
        while state.votes and state.votes[0][0] < cutoff:
            state.votes.popleft()

        recognized = [match for _, match in state.votes if match.recognized]
        counts = Counter(match.employee_id for match in recognized)
        if counts:
            employee_id, count = counts.most_common(1)[0]
            ratio = count / max(1, len(state.votes))
            if count >= self.required_votes and ratio >= self.confirm_ratio:
                matching = [match for match in recognized if match.employee_id == employee_id]
                best = max(matching, key=lambda match: match.similarity)
                average = sum(match.similarity for match in matching) / len(matching)
                state.confirmed = Match(
                    best.employee_id, best.full_name, best.department, average,
                )
                state.expires_at = now + self.ttl_seconds
                return IdentityDecision(state.confirmed, "confirmed")

        if state.confirmed is not None and now <= state.expires_at:
            if candidate.employee_id == state.confirmed.employee_id:
                state.expires_at = now + self.ttl_seconds
            return IdentityDecision(state.confirmed, "cached")
        if candidate.recognized:
            return IdentityDecision(candidate, "verifying")
        return IdentityDecision(candidate, "unknown")

    def remove(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)
