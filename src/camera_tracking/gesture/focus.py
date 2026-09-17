"""Stable foreground selection for camera-facing interactions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from camera_tracking.domain import Track


@dataclass(slots=True)
class ForegroundSelector:
    """Keep focus on the largest person without frame-to-frame ID flicker."""

    switch_area_ratio: float = 1.20
    switch_hold_s: float = 0.50
    lost_grace_s: float = 1.00
    gid: int | None = None
    _challenger_gid: int | None = None
    _challenger_since: float | None = None
    _lost_since: float | None = None

    def update(self, tracks: Iterable[Track], now_s: float) -> int | None:
        visible = {track.track_id: track for track in tracks}
        biggest = max(visible.values(), key=lambda track: track.bbox.area,
                      default=None)
        if self.gid is None:
            if biggest is not None:
                self.gid = biggest.track_id
            return self.gid

        current = visible.get(self.gid)
        if current is None:
            if self._lost_since is None:
                self._lost_since = now_s
            if now_s - self._lost_since < max(0.0, self.lost_grace_s):
                return self.gid
            self.gid = biggest.track_id if biggest is not None else None
            self._reset_transient()
            return self.gid

        self._lost_since = None
        if biggest is None or biggest.track_id == self.gid:
            self._reset_challenger()
            return self.gid

        threshold = current.bbox.area * max(1.0, self.switch_area_ratio)
        if biggest.bbox.area < threshold:
            self._reset_challenger()
            return self.gid

        if self._challenger_gid != biggest.track_id:
            self._challenger_gid = biggest.track_id
            self._challenger_since = now_s
            return self.gid
        since = self._challenger_since if self._challenger_since is not None else now_s
        if now_s - since >= max(0.0, self.switch_hold_s):
            self.gid = biggest.track_id
            self._reset_transient()
        return self.gid

    def remap(self, aliases: Mapping[int, int]) -> None:
        """Follow Global-ID reconciliation without dropping focus state."""
        if self.gid is not None:
            self.gid = _resolve(self.gid, aliases)
        if self._challenger_gid is not None:
            self._challenger_gid = _resolve(self._challenger_gid, aliases)

    def reset(self) -> None:
        self.gid = None
        self._reset_transient()

    def _reset_challenger(self) -> None:
        self._challenger_gid = None
        self._challenger_since = None

    def _reset_transient(self) -> None:
        self._reset_challenger()
        self._lost_since = None


def _resolve(gid: int, aliases: Mapping[int, int]) -> int:
    seen: set[int] = set()
    while gid in aliases and gid not in seen:
        seen.add(gid)
        gid = aliases[gid]
    return gid


__all__ = ["ForegroundSelector"]
