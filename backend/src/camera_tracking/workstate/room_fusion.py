"""Fuse room status from 2 channels + attendance.

Rules:
- Channel A (room): seen -> "Working"; absent past away_grace -> "Away".
- Reappear in channel B (door) after absent in A -> "Out of office"
  (in_room=False). Only confirmed when there is B evidence *after* the
  A-absence started and within leave_confirm_window_s (avoids label
  flicker on detector misses).
- Stable return to A -> "Working" (in_room=True).
- RETURNING from business tracker -> "Returning".
- Overlay shows only Global ID + name (if checked in) + EN label.
  Never shows raw ByteTrack IDs. Labels are plain ASCII on purpose:
  OpenCV Hershey fonts cannot render Vietnamese diacritics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LABEL_WORKING = "Working"
LABEL_AWAY_SEAT = "Away"
LABEL_NEAR_SEAT = "Near seat"
LABEL_OUT_OFFICE = "Out of office"
LABEL_RETURNING = "Returning"
LABEL_UNKNOWN = "Unknown"


@dataclass(slots=True)
class RoomPersonStatus:
    global_id: int
    label: str
    in_room: bool
    display_name: str | None = None
    # True khi transition vua xay ra tai frame nay (de caller log event).
    just_left_office: bool = False
    just_returned: bool = False


@dataclass(slots=True)
class RoomPresenceAggregator:
    """Gop ChannelBusinessTracker A/B + attendance names thanh label hien thi."""

    leave_confirm_window_s: float = 300.0
    # Fallback thực tế: vắng A quá lâu mà không có B (ra bằng cửa khuất)
    # thì vẫn kết luận Out để dashboard khỏi kẹt "Away" cả ngày.
    # 0 = tắt fallback, giữ behavior bảo thủ cũ.
    absent_fallback_s: float = 900.0
    _last_seen_b: dict[int, float] = field(default_factory=dict, init=False)
    _absent_since: dict[int, float] = field(default_factory=dict, init=False)
    _was_out: dict[int, bool] = field(default_factory=dict, init=False)

    def forget(self, gid: int) -> None:
        """Drop per-ID memory (dead ghosts must not linger in the UI)."""
        self._last_seen_b.pop(gid, None)
        self._absent_since.pop(gid, None)
        self._was_out.pop(gid, None)

    def update(
        self,
        now_s: float,
        *,
        present_a: set[int],
        present_b: set[int],
        states_a: dict[int, str],
        states_b: dict[int, str],
        names: dict[int, str],
    ) -> dict[int, RoomPersonStatus]:
        for gid in present_b:
            self._last_seen_b[gid] = now_s
        gids = (
            set(present_a) | set(present_b)
            | set(states_a) | set(states_b)
            | set(self._last_seen_b) | set(self._was_out)
        )
        out: dict[int, RoomPersonStatus] = {}
        for gid in sorted(gids):
            state_a = states_a.get(gid, "UNKNOWN")
            in_a = gid in present_a
            if in_a:
                self._absent_since.pop(gid, None)
            else:
                self._absent_since.setdefault(gid, now_s)
            last_b = self._last_seen_b.get(gid)
            absent_start = self._absent_since.get(gid, now_s)
            # B evidence only counts when it happens after the A-absence
            # started (not a stale sighting) and inside the confirm window.
            seen_b_recently = (
                last_b is not None
                and last_b >= absent_start
                and (now_s - last_b) <= self.leave_confirm_window_s
            )
            was_out = self._was_out.get(gid, False)
            name = names.get(gid)

            just_left = False
            just_back = False
            if in_a:
                if was_out:
                    just_back = True
                # Visible in A, but the position-driven business state still
                # rules: standing elsewhere while visible is Away/Near, not
                # Working. (Presence-only fallback never yields these states
                # together with in_a, so old behavior is preserved there.)
                if state_a == "NEAR_SEAT":
                    label, in_room, out_now = LABEL_NEAR_SEAT, True, False
                elif state_a == "POSSIBLY_OUT" and seen_b_recently:
                    if not was_out:
                        just_left = True
                    label, in_room, out_now = LABEL_OUT_OFFICE, False, True
                elif state_a in ("AWAY_TEMP", "POSSIBLY_OUT"):
                    label, in_room, out_now = LABEL_AWAY_SEAT, True, False
                else:
                    label, in_room, out_now = LABEL_WORKING, True, False
            elif state_a == "RETURNING" or states_b.get(gid) == "RETURNING":
                label, in_room, out_now = LABEL_RETURNING, True, False
            elif state_a == "NEAR_SEAT":
                label, in_room, out_now = LABEL_NEAR_SEAT, True, False
            elif state_a == "AWAY_TEMP":
                label, in_room, out_now = LABEL_AWAY_SEAT, True, False
            elif state_a == "POSSIBLY_OUT" and seen_b_recently:
                if not was_out:
                    just_left = True
                label, in_room, out_now = LABEL_OUT_OFFICE, False, True
            elif state_a == "POSSIBLY_OUT":
                absent_for = now_s - absent_start
                if (
                    self.absent_fallback_s > 0
                    and absent_for >= self.absent_fallback_s
                ):
                    if not was_out:
                        just_left = True
                    label, in_room, out_now = LABEL_OUT_OFFICE, False, True
                else:
                    # Absent long in A but no B evidence yet -> stay conservative:
                    # still away from seat, do not conclude out of office.
                    label, in_room, out_now = LABEL_AWAY_SEAT, True, False
            elif gid in present_b and not in_a and was_out:
                label, in_room, out_now = LABEL_OUT_OFFICE, False, True
            else:
                label, in_room, out_now = LABEL_UNKNOWN, True, False

            self._was_out[gid] = out_now
            out[gid] = RoomPersonStatus(
                global_id=gid, label=label, in_room=in_room,
                display_name=name, just_left_office=just_left,
                just_returned=just_back,
            )
        return out
