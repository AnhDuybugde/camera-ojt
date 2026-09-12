"""Diem danh debounce cho channel B (cua ra vao).

Quy tac:
- Moi global_id giu sliding window cac match gan nhat (timestamp, person_id, score).
- Khi co >= debounce_hits match CUNG person trong window_s va score >= threshold
  -> tick attendance 1 lan/ngay (khong tick 2 lan cung nguoi cung ngay).
- Mat la nhung chua dat debounce -> van nho de hien thi / luu best-shot.
- Khong match ai -> unknown (de store luu crop rieng).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date


@dataclass(slots=True)
class AttendanceRecord:
    day: str  # YYYY-MM-DD
    person_id: str
    display_name: str
    global_id: int
    first_seen_at: float  # now_s cua pipeline (giay tu start)
    wall_time: str  # ISO wall-clock de hien thi/report
    face_score: float


@dataclass(slots=True)
class _WindowHit:
    at_s: float
    person_id: str
    score: float


class FaceAttendanceService:
    def __init__(
        self,
        debounce_hits: int = 3,
        window_s: float = 5.0,
        active_hour_start: int | None = None,
        active_hour_end: int | None = None,
    ) -> None:
        self.debounce_hits = max(1, debounce_hits)
        self.window_s = max(0.0, window_s)
        self.active_hour_start = active_hour_start
        self.active_hour_end = active_hour_end
        self._windows: dict[int, deque[_WindowHit]] = {}
        self._ticked: dict[tuple[str, str], AttendanceRecord] = {}
        self._pending_best: dict[int, tuple[str, float]] = {}  # gid -> (person_id, score)

    # -- chinh --
    def observe(
        self,
        *,
        day: str,
        global_id: int,
        person_id: str | None,
        display_name: str | None,
        score: float,
        now_s: float,
        wall_time_iso: str,
    ) -> AttendanceRecord | None:
        """Nhan 1 match (hoac None neu unknown). Tra record khi VUA tick."""
        if person_id is None:
            return None
        window = self._windows.setdefault(global_id, deque())
        window.append(_WindowHit(at_s=now_s, person_id=person_id, score=score))
        cutoff = now_s - self.window_s
        while window and window[0].at_s < cutoff:
            window.popleft()
        best = self._pending_best.get(global_id)
        if best is None or score > best[1]:
            self._pending_best[guid_safe(global_id)] = (person_id, float(score))
        same = [h for h in window if h.person_id == person_id]
        if len(same) < self.debounce_hits:
            return None
        if not self._in_active_hours(wall_time_iso):
            return None
        key = (day, person_id)
        existing = self._ticked.get(key)
        if existing is not None:
            return None  # da tick hom nay -> khong tick lai
        record = AttendanceRecord(
            day=day,
            person_id=person_id,
            display_name=display_name or person_id,
            global_id=global_id,
            first_seen_at=now_s,
            wall_time=wall_time_iso,
            face_score=float(max(h.score for h in same)),
        )
        self._ticked[key] = record
        return record

    def is_ticked(self, day: str, person_id: str) -> bool:
        return (day, person_id) in self._ticked

    def record_of(self, day: str, person_id: str) -> AttendanceRecord | None:
        return self._ticked.get((day, person_id))

    def records_for_day(self, day: str) -> list[AttendanceRecord]:
        return [r for (d, _), r in self._ticked.items() if d == day]

    def name_of_ticked_global(self, day: str, global_id: int) -> str | None:
        for record in self.records_for_day(day):
            if record.global_id == global_id:
                return record.display_name
        return None

    def preload(self, records: list[AttendanceRecord]) -> None:
        """Nap tick da co tu DB khi restart giua ngay (chong tick lap)."""
        for record in records:
            self._ticked.setdefault((record.day, record.person_id), record)

    # -- phu --
    def _in_active_hours(self, wall_time_iso: str) -> bool:
        if self.active_hour_start is None or self.active_hour_end is None:
            return True
        try:
            hour = int(wall_time_iso[11:13])
        except (ValueError, IndexError):
            return True
        start, end = self.active_hour_start, self.active_hour_end
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end  # khung qua dem


def guid_safe(gid: int) -> int:
    return int(gid)


def today_str(local_date: date | None = None) -> str:
    from datetime import date as _date

    return (local_date or _date.today()).isoformat()
