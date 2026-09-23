"""Fuse room status from 2 channels + attendance. Exactly 5 labels.

Rules (bbox-center image zones):
- Channel B OUT_OF_DOOR (center in R2) -> "Out of door" (in_room=False).
  Fires just_left_office on the False->True edge of _was_out.
- Channel B AT_DOOR (center in R3) -> "At door" (in_room=True).
- Channel A WORKING (center outside R1) -> "Working" (in_room=True).
- Absent from BOTH channels -> "Away"; still absent past
  absent_fallback_s (default 900s = 15 min) -> "Out of door".
  Any sighting in A or B resets the timer.
- Anything undecided -> "Unknown".
- Stable return to A after out (was_out and visible in A) -> just_returned.

Overlay labels are plain ASCII on purpose: OpenCV Hershey fonts cannot
render Vietnamese diacritics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LABEL_WORKING = "Working"
LABEL_AWAY = "Away"
LABEL_AT_DOOR = "At door"
LABEL_OUT_OF_DOOR = "Out of door"
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

    leave_confirm_window_s: float = 300.0  # deprecated, ignored
    # Absent from BOTH channels longer than this -> "Out of door".
    # 0 = disabled (absent stays "Away"). Default 900s = 15 minutes.
    absent_fallback_s: float = 900.0
    _last_seen_b: dict[int, float] = field(default_factory=dict, init=False)
    _last_seen: dict[int, float] = field(default_factory=dict, init=False)
    _absent_since: dict[int, float] = field(default_factory=dict, init=False)
    _ever_seen: set[int] = field(default_factory=set, init=False)
    _was_out: dict[int, bool] = field(default_factory=dict, init=False)

    def forget(self, gid: int) -> None:
        """Drop per-ID memory (dead ghosts must not linger in the UI)."""
        self._last_seen_b.pop(gid, None)
        self._last_seen.pop(gid, None)
        self._absent_since.pop(gid, None)
        self._ever_seen.discard(gid)
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
        for gid in set(present_a) | set(present_b):
            self._ever_seen.add(gid)
            self._last_seen[gid] = now_s
            self._absent_since.pop(gid, None)
        gids = (
            set(present_a) | set(present_b)
            | set(states_a) | set(states_b)
            | set(self._last_seen_b) | set(self._absent_since)
            | set(self._ever_seen) | set(self._was_out)
        )
        out: dict[int, RoomPersonStatus] = {}
        for gid in sorted(gids):
            state_a = states_a.get(gid, "UNKNOWN")
            state_b = states_b.get(gid, "UNKNOWN")
            in_a = gid in present_a
            in_b = gid in present_b
            was_out = self._was_out.get(gid, False)
            name = names.get(gid)
            if not in_a and not in_b:
                # Absence counts from the last live sighting, not from the
                # first absent frame (frames may skip while overloaded).
                self._absent_since.setdefault(
                    gid, self._last_seen.get(gid, now_s))
            absent_for = now_s - self._absent_since.get(gid, now_s)

            just_left = False
            just_back = False
            # Door-out wins outright, even over a simultaneous A sighting
            # (overlap views): R2 means outside the glass.
            if state_b == "OUT_OF_DOOR":
                if not was_out:
                    just_left = True
                label, in_room, out_now = LABEL_OUT_OF_DOOR, False, True
            elif state_b == "AT_DOOR" and (in_b or not in_a):
                label, in_room, out_now = LABEL_AT_DOOR, True, False
                if was_out and in_a:
                    # At the door and visible inside again: treat as return.
                    just_back = True
                    out_now = False
            elif in_a:
                if was_out:
                    just_back = True
                if state_a == "WORKING":
                    label, in_room, out_now = LABEL_WORKING, True, False
                elif state_a == "AT_DOOR":
                    label, in_room, out_now = LABEL_AT_DOOR, True, False
                elif state_a == "OUT_OF_DOOR":
                    if not was_out:
                        just_left = True
                    label, in_room, out_now = LABEL_OUT_OF_DOOR, False, True
                elif state_a == "AWAY":
                    label, in_room, out_now = LABEL_AWAY, True, False
                else:
                    label, in_room, out_now = LABEL_UNKNOWN, True, False
            elif state_a == "AT_DOOR" or (state_b == "AT_DOOR" and not in_b):
                # Lingering door memory while absent: keep showing door
                # until the channel absence timer flips it to AWAY.
                label, in_room, out_now = LABEL_AT_DOOR, True, False
            elif state_a == "AWAY" or state_b == "AWAY":
                label, in_room, out_now = LABEL_AWAY, True, False
            elif gid in self._ever_seen:
                # Channel already pruned this ID (prune_after_s) but it was
                # seen before: still Away, not Unknown. The timer below
                # promotes it to Out of door after absent_fallback_s.
                label, in_room, out_now = LABEL_AWAY, True, False
            else:
                label, in_room, out_now = LABEL_UNKNOWN, True, False

            # Absent from BOTH channels past the fallback window -> Out of
            # door (same terminal state as R2). Fires just_left once.
            # Door states decided above already mean out/door, skip them.
            if (
                not in_a and not in_b
                and label not in (LABEL_OUT_OF_DOOR, LABEL_AT_DOOR)
                and self.absent_fallback_s > 0
                and absent_for >= self.absent_fallback_s
            ):
                if not was_out:
                    just_left = True
                label, in_room, out_now = LABEL_OUT_OF_DOOR, False, True

            self._was_out[gid] = out_now
            out[gid] = RoomPersonStatus(
                global_id=gid, label=label, in_room=in_room,
                display_name=name, just_left_office=just_left,
                just_returned=just_back,
            )
        return out
