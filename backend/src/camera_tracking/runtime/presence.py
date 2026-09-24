from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class PresenceSummary:
    """Canonical occupancy counters with an explicit freshness contract."""

    active_count: int
    logical_count: int
    visible_count: int
    stale_count: int
    stale_after_s: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _canonical_gid(gid: int, aliases: Mapping[int, int]) -> int:
    current = int(gid)
    visited: set[int] = set()
    while current in aliases and current not in visited:
        visited.add(current)
        current = int(aliases[current])
    return current


def summarize_presence(
    people: Sequence[Mapping[str, Any]],
    *,
    aliases: Mapping[int, int] | None = None,
    stale_after_s: float = 120.0,
) -> PresenceSummary:
    """Deduplicate aliases and separate visible, fresh and logical occupancy.

    ``logical_count`` follows the room state machine. ``active_count`` excludes
    records that have not been observed within the configured TTL, preventing a
    missed exit from inflating the live dashboard forever.
    """

    alias_map = aliases or {}
    ttl = max(1.0, float(stale_after_s))
    logical: set[int] = set()
    active: set[int] = set()
    visible: set[int] = set()

    for person in people:
        try:
            gid = _canonical_gid(int(person.get("gid")), alias_map)
        except (TypeError, ValueError):
            continue
        is_visible = bool(person.get("visible"))
        in_room = bool(person.get("in_room"))
        if is_visible:
            visible.add(gid)
        if not in_room:
            continue
        logical.add(gid)
        try:
            age = float(person.get("last_seen_ago_s"))
        except (TypeError, ValueError):
            age = None
        if is_visible or (age is not None and 0.0 <= age <= ttl):
            active.add(gid)

    return PresenceSummary(
        active_count=len(active),
        logical_count=len(logical),
        visible_count=len(visible),
        stale_count=len(logical - active),
        stale_after_s=ttl,
    )


__all__ = ["PresenceSummary", "summarize_presence"]
