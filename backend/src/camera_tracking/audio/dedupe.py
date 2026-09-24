"""Bounded TTL de-duplicator for polled audio events."""

from __future__ import annotations

from collections import OrderedDict


class EventDedupe:
    def __init__(self, ttl_s: float = 3600.0, max_entries: int = 4096) -> None:
        self.ttl_s = max(30.0, float(ttl_s))
        self.max_entries = max(128, int(max_entries))
        self._seen: OrderedDict[str, float] = OrderedDict()

    def check_and_mark(self, event_id: str, now_s: float) -> bool:
        self._purge(now_s)
        event_id = str(event_id or "").strip()
        if not event_id or event_id in self._seen:
            if event_id in self._seen:
                self._seen.move_to_end(event_id)
            return False
        self._seen[event_id] = float(now_s)
        while len(self._seen) > self.max_entries:
            self._seen.popitem(last=False)
        return True

    def _purge(self, now_s: float) -> None:
        cutoff = float(now_s) - self.ttl_s
        while self._seen:
            _event_id, seen_s = next(iter(self._seen.items()))
            if seen_s >= cutoff:
                break
            self._seen.popitem(last=False)

    def __len__(self) -> int:
        return len(self._seen)


__all__ = ["EventDedupe"]
