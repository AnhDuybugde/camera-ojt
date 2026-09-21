"""Event-driven Bé Xinh companion.

Bé Xinh is always ACTIVE, but she is intentionally quiet while people are simply
sitting and working. Speech is triggered only by meaningful events:

* arrival / return;
* deliberate open-palm greeting;
* approach toward the camera;
* standing up after being stationary;
* long stationary water/rest reminders.

There is no random periodic banter. This keeps multi-person offices calm and
prevents the audio queue from becoming stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Iterable, Protocol

from camera_tracking.audio.announcer import (
    PRIORITY_APPROACH,
    PRIORITY_ARRIVAL,
    PRIORITY_GESTURE,
    PRIORITY_PREWARM,
    PRIORITY_REMINDER,
    PRIORITY_STAND,
)


PERSONA_NAME = "Bé Xinh"
GENERIC_ARRIVAL = "Chào nha! Bé Xinh thấy bạn rồi."
GROUP_ARRIVAL = "Chào mọi người nha! Bé Xinh thấy hết rồi."
GENERIC_WAVE = "Hihi, Bé Xinh thấy rồi nha! Chào bạn."
GENERIC_APPROACH = "Ủa, lại gần Bé Xinh hả?"
GENERIC_STAND = "Ơ, đi đâu đó?"
GENERIC_WATER = "Nè, nhớ uống nước nha!"
GROUP_WATER = "Mọi người ngồi lâu rồi đó. Nhớ uống nước nha!"
GENERIC_REST = "Ngồi lâu rồi. Duỗi người chút nha!"


class SpeechSink(Protocol):
    @property
    def enabled(self) -> bool: ...

    @property
    def speaking(self) -> bool: ...

    def say(
        self,
        text: str,
        *,
        key: str | None = None,
        cooldown_s: float = 0.0,
        priority: int = 50,
        expires_s: float | None = 20.0,
        fallback_text: str | None = None,
    ) -> bool: ...

    def prewarm(self, texts: Iterable[str], *, priority: int = 90) -> int: ...

    def status(self) -> dict: ...


@dataclass
class PersonState:
    person_id: str
    display_name: str
    first_seen_s: float
    last_seen_s: float
    present: bool = True
    seen_before: bool = False
    stationary_since_s: float | None = None
    last_arrival_spoken_s: float = float("-inf")
    last_event_spoken_s: float = float("-inf")
    last_water_s: float = float("-inf")
    last_rest_s: float = float("-inf")
    mood: str = "calm"
    global_ids: set[int] = field(default_factory=set)


@dataclass
class _PendingArrival:
    person_id: str
    display_name: str
    created_s: float
    kind: str  # arrival | return


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


class HamyCompanion:
    """Sparse event scheduler + anti-spam policy for a shared office."""

    def __init__(
        self,
        speaker: SpeechSink,
        *,
        reentry_s: float = 45.0,
        presence_timeout_s: float = 12.0,
        arrival_group_window_s: float = 0.25,
        global_event_gap_s: float = 7.0,
        arrival_burst_s: float = 12.0,
        approach_cooldown_s: float = 180.0,
        stand_cooldown_s: float = 180.0,
        water_after_s: float = 2700.0,
        water_repeat_s: float = 3600.0,
        rest_after_s: float = 5400.0,
        rest_repeat_s: float = 5400.0,
    ) -> None:
        self.speaker = speaker
        self.reentry_s = max(10.0, reentry_s)
        self.presence_timeout_s = max(4.0, presence_timeout_s)
        self.arrival_group_window_s = max(0.10, arrival_group_window_s)
        self.global_event_gap_s = max(2.0, global_event_gap_s)
        self.arrival_burst_s = max(5.0, arrival_burst_s)
        self.approach_cooldown_s = max(30.0, approach_cooldown_s)
        self.stand_cooldown_s = max(30.0, stand_cooldown_s)
        self.water_after_s = max(600.0, water_after_s)
        self.water_repeat_s = max(900.0, water_repeat_s)
        self.rest_after_s = max(self.water_after_s, rest_after_s)
        self.rest_repeat_s = max(1800.0, rest_repeat_s)

        self._people: dict[str, PersonState] = {}
        self._gid_to_person: dict[int, str] = {}
        self._pending_arrivals: dict[str, _PendingArrival] = {}
        self._last_non_gesture_s = float("-inf")
        self._last_arrival_batch_s = float("-inf")
        self._last_line = ""
        self._last_event = "idle"

    @classmethod
    def from_env(cls, speaker: SpeechSink) -> "HamyCompanion":
        return cls(
            speaker,
            reentry_s=_env_float("HAMY_REENTRY_SECONDS", 45.0, 10.0),
            presence_timeout_s=_env_float(
                "HAMY_PRESENCE_TIMEOUT_SECONDS", 12.0, 4.0
            ),
            arrival_group_window_s=_env_float(
                "HAMY_ARRIVAL_GROUP_WINDOW_SECONDS", 0.25, 0.10
            ),
            global_event_gap_s=_env_float(
                "HAMY_GLOBAL_EVENT_GAP_SECONDS", 7.0, 2.0
            ),
            arrival_burst_s=_env_float(
                "HAMY_ARRIVAL_BURST_SECONDS", 12.0, 5.0
            ),
            approach_cooldown_s=_env_float(
                "HAMY_APPROACH_COOLDOWN_SECONDS", 180.0, 30.0
            ),
            stand_cooldown_s=_env_float(
                "HAMY_STAND_COOLDOWN_SECONDS", 180.0, 30.0
            ),
            water_after_s=_env_float("HAMY_WATER_AFTER_SECONDS", 2700.0, 600.0),
            water_repeat_s=_env_float(
                "HAMY_WATER_REPEAT_SECONDS", 3600.0, 900.0
            ),
            rest_after_s=_env_float("HAMY_REST_AFTER_SECONDS", 5400.0, 1200.0),
            rest_repeat_s=_env_float(
                "HAMY_REST_REPEAT_SECONDS", 5400.0, 1800.0
            ),
        )

    @property
    def active(self) -> bool:
        return bool(getattr(self.speaker, "enabled", True))

    def prewarm_people(self, people: Iterable[tuple[str, str]]) -> None:
        """Do not synthesize while live tracking is starting.

        The production worker is cache-first so YOLO/InsightFace keep all GPU
        headroom. Build the complete speech cache offline with
        scripts/prewarm_hamy.py before starting run_workstate.py.
        """
        del people
        return None

    def identify(
        self,
        *,
        global_id: int,
        person_id: str,
        display_name: str,
        now_s: float,
    ) -> None:
        """Bind a Global ID immediately when face recognition succeeds.

        This is called directly in the face-result block, not later during
        attendance/fusion, so greeting scheduling starts as soon as identity is
        known.
        """
        person_id = str(person_id or "").strip()
        name = self._clean(display_name)
        if not person_id or not name:
            return

        self._gid_to_person[int(global_id)] = person_id
        state = self._people.get(person_id)
        if state is None:
            state = PersonState(
                person_id=person_id,
                display_name=name,
                first_seen_s=now_s,
                last_seen_s=now_s,
            )
            state.global_ids.add(int(global_id))
            self._people[person_id] = state
            self._queue_arrival(state, now_s, kind="arrival")
            return

        gap_s = max(0.0, now_s - state.last_seen_s)
        was_absent = not state.present
        state.display_name = name
        state.last_seen_s = now_s
        state.present = True
        state.global_ids.add(int(global_id))

        if was_absent and gap_s >= self.reentry_s:
            self._queue_arrival(state, now_s, kind="return")

    def observe_presence(
        self,
        *,
        person_id: str,
        display_name: str,
        now_s: float,
        stationary_since_s: float | None = None,
        global_id: int | None = None,
    ) -> None:
        """Heartbeat a known person from cheap Global-ID tracking."""
        person_id = str(person_id or "").strip()
        name = self._clean(display_name)
        if not person_id or not name:
            return

        state = self._people.get(person_id)
        if state is None:
            state = PersonState(
                person_id=person_id,
                display_name=name,
                first_seen_s=now_s,
                last_seen_s=now_s,
            )
            self._people[person_id] = state
            self._queue_arrival(state, now_s, kind="arrival")
        else:
            gap_s = max(0.0, now_s - state.last_seen_s)
            was_absent = not state.present
            state.display_name = name
            state.last_seen_s = now_s
            state.present = True
            if was_absent and gap_s >= self.reentry_s:
                self._queue_arrival(state, now_s, kind="return")

        if global_id is not None:
            gid = int(global_id)
            state.global_ids.add(gid)
            self._gid_to_person[gid] = person_id
        state.stationary_since_s = stationary_since_s

    # Compatibility with the previous always-on patch.
    def see_person(self, person_id: str, display_name: str, now_s: float) -> None:
        self.observe_presence(
            person_id=person_id,
            display_name=display_name,
            now_s=now_s,
        )

    def wave(
        self,
        *,
        now_s: float,
        person_id: str | None = None,
        display_name: str | None = None,
        global_id: int | None = None,
    ) -> bool:
        """Highest-priority response to a deliberate open-palm greeting."""
        if person_id is None and global_id is not None:
            person_id = self._gid_to_person.get(int(global_id))
        state = self._people.get(person_id or "") if person_id else None
        name = self._clean(display_name or (state.display_name if state else ""))

        if name and person_id:
            text = self._wave_text(name)
            key = f"wave:{person_id}"
        else:
            text = GENERIC_WAVE
            key = f"wave:gid:{global_id or 'unknown'}"

        accepted = self.speaker.say(
            text,
            key=key,
            cooldown_s=10.0,
            priority=PRIORITY_GESTURE,
            expires_s=8.0,
            fallback_text=GENERIC_WAVE,
        )
        if accepted:
            self._note_spoken("wave", text, now_s, state, non_gesture=False)
        return accepted

    def motion_event(
        self,
        *,
        kind: str,
        now_s: float,
        global_id: int,
        person_id: str | None,
        display_name: str | None,
    ) -> bool:
        """Respond to APPROACH / STAND_UP only when it adds value."""
        if person_id is None:
            person_id = self._gid_to_person.get(int(global_id))
        state = self._people.get(person_id or "") if person_id else None
        name = self._clean(display_name or (state.display_name if state else ""))
        kind = str(kind).upper()

        # Sitting/working has no event and therefore produces no speech.
        if kind not in {"APPROACH", "STAND_UP"}:
            return False

        # If we just greeted this arrival, do not immediately follow it with a
        # second sentence about approaching/standing.
        if state and now_s - state.last_arrival_spoken_s < 30.0:
            return False
        if now_s - self._last_non_gesture_s < self.global_event_gap_s:
            return False

        if kind == "APPROACH":
            text = (
                f"Ủa, {name} lại gần Bé Xinh hả?"
                if name
                else "Ủa, lại gần Bé Xinh hả?"
            )
            key = f"approach:{person_id or global_id}"
            cooldown = self.approach_cooldown_s
            priority = PRIORITY_APPROACH
            fallback = GENERIC_APPROACH
            mood = "playful"
        else:
            if not name or not person_id:
                return False
            text = f"Ơ, {name} đi đâu đó?"
            key = f"stand:{person_id}"
            cooldown = self.stand_cooldown_s
            priority = PRIORITY_STAND
            fallback = GENERIC_STAND
            mood = "curious"

        accepted = self.speaker.say(
            text,
            key=key,
            cooldown_s=cooldown,
            priority=priority,
            expires_s=10.0,
            fallback_text=fallback,
        )
        if accepted:
            if state:
                state.mood = mood
            self._note_spoken(kind.lower(), text, now_s, state, non_gesture=True)
        return accepted

    def tick(self, now_s: float) -> None:
        """Advance absence, flush grouped arrivals, and issue rare reminders."""
        for state in self._people.values():
            if state.present and now_s - state.last_seen_s >= self.presence_timeout_s:
                state.present = False
                state.stationary_since_s = None
                state.seen_before = True

        self._flush_arrivals(now_s)

        # Never stack a reminder immediately after a more important event.
        if now_s - self._last_non_gesture_s < self.global_event_gap_s:
            return

        stationary = [
            state
            for state in self._people.values()
            if state.present and state.stationary_since_s is not None
        ]
        if not stationary:
            return

        water_due = [
            state
            for state in stationary
            if now_s - state.stationary_since_s >= self.water_after_s
            and now_s - state.last_water_s >= self.water_repeat_s
        ]
        if water_due:
            if len(water_due) >= 2:
                text = GROUP_WATER
                key = "water:group"
                state_for_note = None
                for state in water_due:
                    state.last_water_s = now_s
                    state.mood = "caring"
            else:
                state_for_note = water_due[0]
                text = f"{state_for_note.display_name} ơi, uống miếng nước đi nha!"
                key = f"water:{state_for_note.person_id}"
                state_for_note.last_water_s = now_s
                state_for_note.mood = "caring"

            accepted = self.speaker.say(
                text,
                key=key,
                cooldown_s=self.water_repeat_s,
                priority=PRIORITY_REMINDER,
                expires_s=30.0,
                fallback_text=(GROUP_WATER if len(water_due) >= 2 else GENERIC_WATER),
            )
            if accepted:
                self._note_spoken(
                    "water", text, now_s, state_for_note, non_gesture=True
                )
            return

        rest_due = [
            state
            for state in stationary
            if now_s - state.stationary_since_s >= self.rest_after_s
            and now_s - state.last_rest_s >= self.rest_repeat_s
        ]
        if rest_due:
            state = rest_due[0]
            text = f"{state.display_name} ơi, ngồi lâu rồi. Duỗi người chút nha!"
            accepted = self.speaker.say(
                text,
                key=f"rest:{state.person_id}",
                cooldown_s=self.rest_repeat_s,
                priority=PRIORITY_REMINDER,
                expires_s=30.0,
                fallback_text=GENERIC_REST,
            )
            if accepted:
                state.last_rest_s = now_s
                state.mood = "caring"
                self._note_spoken("rest", text, now_s, state, non_gesture=True)

    def status(self, now_s: float | None = None) -> dict:
        del now_s
        present = [state for state in self._people.values() if state.present]
        speaker_status = self.speaker.status()
        return {
            "active": self.active,
            "mode": "event_driven",
            "muted": bool(speaker_status.get("muted")),
            "speaking": bool(speaker_status.get("speaking")),
            "ready": bool(speaker_status.get("ready")),
            "queue_depth": int(speaker_status.get("queue_depth", 0)),
            "present_count": len(present),
            "people": [state.display_name for state in present],
            "moods": {state.person_id: state.mood for state in present},
            "last_event": self._last_event,
            "last_line": self._last_line,
            "persona": PERSONA_NAME,
        }

    def forget_global_ids(self, alive_global_ids: set[int]) -> None:
        for gid in [gid for gid in self._gid_to_person if gid not in alive_global_ids]:
            self._gid_to_person.pop(gid, None)
        for state in self._people.values():
            state.global_ids.intersection_update(alive_global_ids)

    def _queue_arrival(self, state: PersonState, now_s: float, *, kind: str) -> None:
        existing = self._pending_arrivals.get(state.person_id)
        if existing is not None and now_s - existing.created_s < self.reentry_s:
            return
        self._pending_arrivals[state.person_id] = _PendingArrival(
            person_id=state.person_id,
            display_name=state.display_name,
            created_s=now_s,
            kind=kind,
        )

    def _flush_arrivals(self, now_s: float) -> None:
        if not self._pending_arrivals:
            return
        earliest = min(item.created_s for item in self._pending_arrivals.values())
        if now_s - earliest < self.arrival_group_window_s:
            return

        arrivals = list(self._pending_arrivals.values())
        self._pending_arrivals.clear()

        # Face matches for 3-4 simultaneous arrivals can finish a few seconds
        # apart because InsightFace is intentionally serial. After one arrival
        # greeting starts, suppress the rest of that burst instead of building
        # an audio backlog that speaks when people are already at their desks.
        if now_s - self._last_arrival_batch_s < self.arrival_burst_s:
            for item in arrivals:
                state = self._people.get(item.person_id)
                if state:
                    state.last_arrival_spoken_s = now_s
            return

        # Group simultaneous arrivals into one clean sentence. This is the main
        # anti-overlap rule for 3-4 people entering together.
        if len(arrivals) >= 2:
            accepted = self.speaker.say(
                GROUP_ARRIVAL,
                key="arrival:group",
                cooldown_s=15.0,
                priority=PRIORITY_ARRIVAL,
                expires_s=10.0,
                fallback_text=GENERIC_ARRIVAL,
            )
            if accepted:
                for item in arrivals:
                    state = self._people.get(item.person_id)
                    if state:
                        state.last_arrival_spoken_s = now_s
                        state.mood = "happy"
                self._last_arrival_batch_s = now_s
                self._note_spoken(
                    "group_arrival", GROUP_ARRIVAL, now_s, None, non_gesture=True
                )
            return

        item = arrivals[0]
        state = self._people.get(item.person_id)
        if state is None:
            return
        text = (
            self._return_text(state.display_name)
            if item.kind == "return"
            else self._arrival_text(state.display_name)
        )
        accepted = self.speaker.say(
            text,
            key=f"{item.kind}:{item.person_id}",
            cooldown_s=self.reentry_s,
            priority=PRIORITY_ARRIVAL,
            expires_s=12.0,
            fallback_text=GENERIC_ARRIVAL,
        )
        if accepted:
            state.last_arrival_spoken_s = now_s
            state.mood = "happy"
            self._last_arrival_batch_s = now_s
            self._note_spoken(item.kind, text, now_s, state, non_gesture=True)

    def _note_spoken(
        self,
        event: str,
        text: str,
        now_s: float,
        state: PersonState | None,
        *,
        non_gesture: bool,
    ) -> None:
        self._last_event = event
        self._last_line = text
        if non_gesture:
            self._last_non_gesture_s = now_s
        if state is not None:
            state.last_event_spoken_s = now_s
        print(f"[Bé Xinh/{event}] {text}")

    @staticmethod
    def _clean(value: str | None) -> str:
        return " ".join(str(value or "").split())

    @staticmethod
    def _arrival_text(name: str) -> str:
        return f"{name} tới rồi nè! Bé Xinh chào nha."

    @staticmethod
    def _return_text(name: str) -> str:
        return f"{name} quay lại rồi nè! Bé Xinh chào nha."

    @staticmethod
    def _wave_text(name: str) -> str:
        return f"Hihi, {name} chào Bé Xinh hả? Chào nha."


__all__ = [
    "HamyCompanion",
    "GENERIC_ARRIVAL",
    "GROUP_ARRIVAL",
    "GENERIC_WAVE",
    "GENERIC_APPROACH",
    "GENERIC_STAND",
    "GENERIC_WATER",
    "GROUP_WATER",
    "GENERIC_REST",
]
