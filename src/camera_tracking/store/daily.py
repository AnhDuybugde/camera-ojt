"""DailyStateCache: luat chong spam write DB theo yeu cau.

- attendance: 1 lan / ngay / nguoi.
- room_status_daily: update khi label doi; rieng flip in_room toi da
  1 lan / gio (inroom_min_interval_s). Cac flip trung gian van duoc goi y
  ghi vao room_events (append-only) de audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def throttle_ok(last_write_s: float | None, now_s: float, min_interval_s: float) -> bool:
    if last_write_s is None:
        return True
    return (now_s - last_write_s) >= min_interval_s


@dataclass(slots=True)
class RoomStatusRow:
    day: str
    global_id: int
    label: str
    in_room: bool
    last_write_s: float | None = None
    last_inroom_flip_s: float | None = None
    last_leave_at: str | None = None
    last_enter_at: str | None = None


@dataclass(slots=True)
class DailyStateCache:
    inroom_min_interval_s: float = 3600.0
    heartbeat_interval_s: float = 30.0
    _attendance_done: set[tuple[str, str]] = field(default_factory=set, init=False)
    _status: dict[tuple[str, int], RoomStatusRow] = field(default_factory=dict, init=False)

    # -- attendance: True nghia la "duoc phep ghi DB lan nay" --
    def attendance_should_write(self, day: str, person_id: str) -> bool:
        key = (day, person_id)
        if key in self._attendance_done:
            return False
        self._attendance_done.add(key)
        return True

    def preload_attendance(self, day: str, person_ids: list[str]) -> None:
        for pid in person_ids:
            self._attendance_done.add((day, pid))

    # -- room status --
    def status_should_write(
        self,
        *,
        day: str,
        global_id: int,
        label: str,
        in_room: bool,
        now_s: float,
    ) -> tuple[bool, bool]:
        """Tra (write_status_row, log_event).

        - label doi -> write row (tru flip in_room bi throttle thi van log event).
        - flip in_room trong < min_interval -> khong write row, chi log event.
        """
        key = (day, global_id)
        row = self._status.get(key)
        if row is None:
            self._status[key] = RoomStatusRow(
                day=day, global_id=global_id, label=label, in_room=in_room,
                last_write_s=now_s,
                last_inroom_flip_s=now_s,
            )
            return True, True
        label_changed = row.label != label
        inroom_flipped = row.in_room != in_room
        if not label_changed and not inroom_flipped:
            if now_s - row.last_write_s >= self.heartbeat_interval_s:
                row.last_write_s = now_s
                return True, False
            return False, False
        if inroom_flipped and not throttle_ok(
            row.last_inroom_flip_s, now_s, self.inroom_min_interval_s
        ):
            # Coalesce: giu row cu, chi log event de khong tran traffic.
            return False, True
        row.label = label
        row.in_room = in_room
        row.last_write_s = now_s
        if inroom_flipped:
            row.last_inroom_flip_s = now_s
        return True, True

    def note_leave_enter(
        self, day: str, global_id: int, *, leave_at: str | None = None,
        enter_at: str | None = None,
    ) -> None:
        row = self._status.get((day, global_id))
        if row is None:
            return
        if leave_at is not None:
            row.last_leave_at = leave_at
        if enter_at is not None:
            row.last_enter_at = enter_at

    def status_of(self, day: str, global_id: int) -> RoomStatusRow | None:
        return self._status.get((day, global_id))
