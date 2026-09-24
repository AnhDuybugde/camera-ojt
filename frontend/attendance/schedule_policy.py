"""Single source for weekly registration windows and completion rules."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Iterable, Mapping


SESSIONS = frozenset({"MORNING", "AFTERNOON"})
STATUSES = frozenset({"ON", "WFH", "OFF"})


def monday(value: date) -> date:
    return value - timedelta(days=value.weekday())


def get_next_week_range(now: date | datetime | None = None) -> tuple[date, date]:
    day = (now or datetime.now()).date() if isinstance(now, datetime) or now is None else now
    start = monday(day) + timedelta(days=7)
    return start, start + timedelta(days=4)


def get_schedule_deadline(now: date | datetime | None = None) -> datetime:
    day = (now or datetime.now()).date() if isinstance(now, datetime) or now is None else now
    friday = monday(day) + timedelta(days=4)
    return datetime.combine(friday, time(23, 59, 59))


def can_employee_edit_schedule(work_date: date | str, now: datetime | None = None) -> bool:
    current = now or datetime.now()
    target = date.fromisoformat(work_date) if isinstance(work_date, str) else work_date
    start, end = get_next_week_range(current)
    return current.weekday() <= 4 and start <= target <= end


@dataclass(frozen=True)
class WeekCompletion:
    registered_days: int
    selected_slots: int = 0
    total_days: int = 5

    @property
    def status(self) -> str:
        if self.selected_slots == 0:
            return "UNREGISTERED"
        if self.registered_days == self.total_days:
            return "COMPLETE"
        return "INCOMPLETE"


def get_week_schedule_completion(
    rows: Iterable[Mapping], employee_id: str, week_start: date,
) -> WeekCompletion:
    """A day is complete only when both existing morning/afternoon cells are set."""
    start = monday(week_start)
    end = start + timedelta(days=4)
    sessions_by_day: dict[date, set[str]] = {}
    for row in rows:
        if row["employee_id"] != employee_id or row["work_status"] not in STATUSES:
            continue
        day = date.fromisoformat(row["work_date"])
        if start <= day <= end and day.weekday() < 5:
            sessions_by_day.setdefault(day, set()).add(row["work_session"])
    complete = sum(sessions >= SESSIONS for sessions in sessions_by_day.values())
    selected = sum(len(sessions) for sessions in sessions_by_day.values())
    return WeekCompletion(complete, selected)
