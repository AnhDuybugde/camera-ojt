"""Event-driven Bé Xinh companion.

Bé Xinh is always ACTIVE, but she is intentionally quiet while people are simply
sitting and working. Speech is triggered only by meaningful events:

* arrival / return;
* deliberate open-palm greeting;
* approach toward the camera;
* standing up after being stationary;
* long stationary water/rest reminders;
* friendly, rotating check-ins after a long period without interaction.

Proactive lines are state-driven rather than random. Per-person and global
cooldowns keep a shared office calm and prevent stale audio queues.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from camera_tracking.audio.announcer import (
    PRIORITY_APPROACH,
    PRIORITY_ARRIVAL,
    PRIORITY_GESTURE,
    PRIORITY_REMINDER,
    PRIORITY_STAND,
)

PERSONA_NAME = "Bé Xinh"
GENERIC_ARRIVAL = "Chào nha! Bé Xinh thấy bạn rồi."
GROUP_ARRIVAL = "Chào mọi người nha! Bé Xinh thấy hết rồi."
GENERIC_WAVE = "Hihi, Bé Xinh thấy năm ngón tay rồi nha! Chào bạn."
GENERIC_APPROACH = "Ủa, lại gần Bé Xinh hả?"
GENERIC_STAND = "Ơ, đi đâu đó?"
GENERIC_WATER = "Nè, nhớ uống nước nha!"
GROUP_WATER = "Mọi người ngồi lâu rồi đó. Nhớ uống nước nha!"
GENERIC_REST = "Ngồi lâu rồi. Duỗi người chút nha!"
GENERIC_BANTER = "Bé Xinh vẫn ở đây nè. Giơ năm ngón tay chào Bé Xinh một cái nha!"
GROUP_BANTER = "Cả nhà tập trung quá trời. Nghỉ mắt một chút rồi giơ tay chào Bé Xinh nha!"

ARRIVAL_TEMPLATES = (
    "{name} tới rồi nè! Bé Xinh chào nha.",
    "Chào {name} nha! Bé Xinh vui vì gặp {name} nè.",
    "Hi {name}! Bé Xinh thấy {name} rồi nha.",
    "{name} tới rồi! Chúc {name} một ngày thật vui nha.",
    "Xin chào {name}! Bé Xinh gửi {name} một chiếc high-five từ xa nè.",
)
RETURN_TEMPLATES = (
    "{name} quay lại rồi nè! Bé Xinh chào nha.",
    "{name} quay lại rồi nè! Bé Xinh chào {name} nha.",
    "Chào mừng {name} quay lại! Bé Xinh nhớ {name} đó nha.",
    "A, {name} đây rồi! Bé Xinh chào {name} nè.",
)
NEAR_GREETING_TEMPLATES = (
    "{name} tới rồi nè! Bé Xinh chào nha.",
    "Chào {name} nha! Bé Xinh thấy {name} đến gần rồi nè.",
    "Hi {name}! Lại gần Bé Xinh là được chào liền nha.",
    "{name} ơi, Bé Xinh thấy {name} rồi. Chào một cái thật dễ thương nè!",
)
PERSONAL_WAVE_TEMPLATES = (
    "Hihi, {name} chào Bé Xinh hả? Chào nha.",
    "Hihi, Bé Xinh thấy {name} vẫy tay rồi nè. Chào {name} nha!",
    "{name} ơi, high-five từ xa nè! Bé Xinh chào {name}.",
    "Chào {name}! Bé Xinh bắt được tín hiệu năm ngón tay rồi nha.",
    "Bé Xinh thấy rồi nè, chào {name} dễ thương nha!",
)
GENERIC_WAVE_LINES = (
    GENERIC_WAVE,
    "Bắt được tín hiệu high-five rồi! Bé Xinh chào lại nè.",
    "Xin chào bạn nha! Bé Xinh thấy bàn tay rồi đó.",
)
GENERIC_WATER_LINES = (
    GENERIC_WATER,
    "Tập trung tốt lắm, nhưng nhớ uống nước nữa nha.",
    "Bé Xinh nhắc nhẹ: tới giờ nạp nước rồi đó!",
)
GENERIC_REST_LINES = (
    GENERIC_REST,
    "Nghỉ mắt hai mươi giây nha, nhìn xa một chút rồi làm tiếp.",
    "Bé Xinh thấy mọi người chăm chỉ quá. Thả lỏng vai một chút nha!",
)
BANTER_LINES = (
    GENERIC_BANTER,
    "Tập trung quá trời luôn. Chớp mắt, thả lỏng vai rồi mình làm tiếp nha!",
    "Bé Xinh thấy mọi người chăm chỉ ghê. Cho Bé Xinh một cái high-five từ xa nha!",
    "Không khí nghiêm túc quá nè. Cười một cái rồi làm tiếp nha!",
)
GROUP_BANTER_LINES = (
    GROUP_BANTER,
    "Mọi người làm việc chăm ghê! Cùng thả lỏng vai một chút nha.",
    "Bé Xinh ngồi đây nãy giờ mà chưa ai high-five hết á. Ai chào Bé Xinh nào!",
    "Cả nhà ơi, nhìn xa hai mươi giây cho mắt nghỉ rồi mình tiếp tục nha.",
)


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
    last_banter_s: float = float("-inf")
    last_interaction_s: float | None = None
    near_camera: bool = False
    estimated_distance_m: float | None = None
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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class HamyCompanion:
    """Sparse event scheduler + anti-spam policy for a shared office."""

    def __init__(
        self,
        speaker: SpeechSink,
        *,
        reentry_s: float = 45.0,
        presence_timeout_s: float = 12.0,
        arrival_delay_s: float = 5.0,
        arrival_group_window_s: float = 0.25,
        global_event_gap_s: float = 7.0,
        wave_global_gap_s: float = 15.0,
        arrival_burst_s: float = 12.0,
        approach_cooldown_s: float = 180.0,
        stand_cooldown_s: float = 180.0,
        water_after_s: float = 2700.0,
        water_repeat_s: float = 3600.0,
        rest_after_s: float = 5400.0,
        rest_repeat_s: float = 5400.0,
        banter_after_s: float = 1200.0,
        banter_repeat_s: float = 1800.0,
        greet_only_when_near: bool = False,
        auto_arrival_from_presence: bool = True,
    ) -> None:
        self.speaker = speaker
        self.reentry_s = max(10.0, reentry_s)
        self.presence_timeout_s = max(4.0, presence_timeout_s)
        self.arrival_delay_s = max(0.0, arrival_delay_s)
        self.arrival_group_window_s = max(0.10, arrival_group_window_s)
        self.global_event_gap_s = max(2.0, global_event_gap_s)
        self.wave_global_gap_s = max(5.0, wave_global_gap_s)
        self.arrival_burst_s = max(5.0, arrival_burst_s)
        self.approach_cooldown_s = max(30.0, approach_cooldown_s)
        self.stand_cooldown_s = max(30.0, stand_cooldown_s)
        self.water_after_s = max(600.0, water_after_s)
        self.water_repeat_s = max(900.0, water_repeat_s)
        self.rest_after_s = max(self.water_after_s, rest_after_s)
        self.rest_repeat_s = max(1800.0, rest_repeat_s)
        self.banter_after_s = max(300.0, banter_after_s)
        self.banter_repeat_s = max(600.0, banter_repeat_s)
        self.greet_only_when_near = bool(greet_only_when_near)
        self.auto_arrival_from_presence = bool(auto_arrival_from_presence)

        self._people: dict[str, PersonState] = {}
        self._gid_to_person: dict[int, str] = {}
        self._pending_arrivals: dict[str, _PendingArrival] = {}
        self._last_non_gesture_s = float("-inf")
        self._last_arrival_batch_s = float("-inf")
        self._last_wave_s = float("-inf")
        self._recent_wave_gids: dict[int, float] = {}
        self._last_line = ""
        self._last_event = "idle"
        self._line_counters: dict[str, int] = {}

    @classmethod
    def from_env(cls, speaker: SpeechSink) -> HamyCompanion:
        return cls(
            speaker,
            reentry_s=_env_float("HAMY_REENTRY_SECONDS", 45.0, 10.0),
            presence_timeout_s=_env_float(
                "HAMY_PRESENCE_TIMEOUT_SECONDS", 12.0, 4.0
            ),
            arrival_delay_s=_env_float(
                "HAMY_ARRIVAL_DELAY_SECONDS", 5.0, 0.0
            ),
            arrival_group_window_s=_env_float(
                "HAMY_ARRIVAL_GROUP_WINDOW_SECONDS", 0.25, 0.10
            ),
            global_event_gap_s=_env_float(
                "HAMY_GLOBAL_EVENT_GAP_SECONDS", 7.0, 2.0
            ),
            wave_global_gap_s=_env_float(
                "HAMY_WAVE_GLOBAL_GAP_SECONDS", 15.0, 5.0
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
            banter_after_s=_env_float(
                "HAMY_BANTER_AFTER_SECONDS", 1200.0, 300.0
            ),
            banter_repeat_s=_env_float(
                "HAMY_BANTER_REPEAT_SECONDS", 1800.0, 600.0
            ),
            greet_only_when_near=_env_bool(
                "HAMY_GREET_ONLY_WHEN_NEAR", True
            ),
            auto_arrival_from_presence=_env_bool(
                "HAMY_AUTO_ARRIVAL_FROM_PRESENCE", True
            ),
        )

    @property
    def active(self) -> bool:
        return bool(getattr(self.speaker, "enabled", True))

    def set_auto_arrival_from_presence(self, enabled: bool) -> None:
        """Switch legacy presence greetings without losing other companion state.

        ``audio_events`` owns arrivals in Zone Mode. Clearing pending presence
        greetings here prevents a face heartbeat racing a door event and
        producing two sentences.
        """
        enabled = bool(enabled)
        if self.auto_arrival_from_presence == enabled:
            return
        self.auto_arrival_from_presence = enabled
        if not enabled:
            self._pending_arrivals.clear()

    def prewarm_people(self, people: Iterable[tuple[str, str]]) -> None:
        """Do not synthesize while live tracking is starting.

        The production worker is cache-first so YOLO/InsightFace keep all GPU
        headroom. Build the complete speech cache offline with
        scripts/prewarm_hamy.py before starting run_workstate.py.
        """
        del people

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
            if (
                self.auto_arrival_from_presence
                and not self._recent_wave_for_gid(global_id, now_s)
            ):
                self._queue_arrival(state, now_s, kind="arrival")
            return

        gap_s = max(0.0, now_s - state.last_seen_s)
        was_absent = not state.present
        state.display_name = name
        state.last_seen_s = now_s
        state.present = True
        state.global_ids.add(int(global_id))

        if (
            self.auto_arrival_from_presence
            and was_absent
            and gap_s >= self.reentry_s
        ):
            self._queue_arrival(state, now_s, kind="return")

    def observe_presence(
        self,
        *,
        person_id: str,
        display_name: str,
        now_s: float,
        stationary_since_s: float | None = None,
        global_id: int | None = None,
        near_camera: bool | None = None,
        estimated_distance_m: float | None = None,
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
            if self.auto_arrival_from_presence and (
                global_id is None or not self._recent_wave_for_gid(global_id, now_s)
            ):
                self._queue_arrival(state, now_s, kind="arrival")
        else:
            gap_s = max(0.0, now_s - state.last_seen_s)
            was_absent = not state.present
            state.display_name = name
            state.last_seen_s = now_s
            state.present = True
            if (
                self.auto_arrival_from_presence
                and was_absent
                and gap_s >= self.reentry_s
            ):
                self._queue_arrival(state, now_s, kind="return")

        if global_id is not None:
            gid = int(global_id)
            state.global_ids.add(gid)
            self._gid_to_person[gid] = person_id
        state.stationary_since_s = stationary_since_s
        if near_camera is not None:
            state.near_camera = bool(near_camera)
        state.estimated_distance_m = estimated_distance_m

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
        if now_s - self._last_wave_s < self.wave_global_gap_s:
            return False
        if person_id is None and global_id is not None:
            person_id = self._gid_to_person.get(int(global_id))
        state = self._people.get(person_id or "") if person_id else None
        name = self._clean(display_name or (state.display_name if state else ""))

        if name and person_id:
            text = self._pick_speakable_line(
                f"wave-line:{person_id}",
                tuple(line.format(name=name) for line in PERSONAL_WAVE_TEMPLATES),
            )
            key = f"wave:{person_id}"
            fallback = text
        else:
            text = self._pick_line("wave-line:generic", GENERIC_WAVE_LINES)
            key = f"wave:gid:{global_id or 'unknown'}"
            fallback = GENERIC_WAVE

        accepted = self.speaker.say(
            text,
            key=key,
            cooldown_s=10.0,
            priority=PRIORITY_GESTURE,
            expires_s=8.0,
            fallback_text=fallback,
        )
        if accepted:
            self._last_wave_s = now_s
            if global_id is not None:
                self._recent_wave_gids[int(global_id)] = now_s
            if state is not None:
                state.last_interaction_s = now_s
                state.last_arrival_spoken_s = now_s
                state.mood = "happy"
                self._pending_arrivals.pop(state.person_id, None)
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
        # In proximity mode, APPROACH is only a motion hint. Speech waits for
        # the calibrated near-distance gate so a distant passer-by is quiet.
        if kind == "APPROACH" and self.greet_only_when_near:
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
                state.near_camera = False
                state.estimated_distance_m = None
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
                text = self._pick_line(
                    f"water-line:{state_for_note.person_id}",
                    (
                        f"{state_for_note.display_name} ơi, uống miếng nước đi nha!",
                        *GENERIC_WATER_LINES,
                    ),
                )
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
            text = self._pick_line(
                f"rest-line:{state.person_id}",
                (
                    f"{state.display_name} ơi, ngồi lâu rồi. Duỗi người chút nha!",
                    *GENERIC_REST_LINES,
                ),
            )
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
            return

        banter_due = [
            state
            for state in stationary
            if now_s
            - max(
                (
                    state.stationary_since_s
                    if state.stationary_since_s is not None
                    else now_s
                ),
                (
                    state.last_interaction_s
                    if state.last_interaction_s is not None
                    else state.first_seen_s
                ),
            )
            >= self.banter_after_s
            and now_s - state.last_banter_s >= self.banter_repeat_s
        ]
        if banter_due:
            if len(banter_due) >= 2:
                text = self._pick_line("banter-line:group", GROUP_BANTER_LINES)
                key = "banter:group"
                state_for_note = None
                for state in banter_due:
                    state.last_banter_s = now_s
                    state.mood = "playful"
            else:
                state_for_note = banter_due[0]
                text = self._pick_line(
                    f"banter-line:{state_for_note.person_id}",
                    BANTER_LINES,
                )
                key = f"banter:{state_for_note.person_id}"
                state_for_note.last_banter_s = now_s
                state_for_note.mood = "playful"

            accepted = self.speaker.say(
                text,
                key=key,
                cooldown_s=self.banter_repeat_s,
                priority=PRIORITY_REMINDER,
                expires_s=30.0,
                fallback_text=(
                    GROUP_BANTER if len(banter_due) >= 2 else GENERIC_BANTER
                ),
            )
            if accepted:
                self._note_spoken(
                    "banter", text, now_s, state_for_note, non_gesture=True
                )

    def status(self, now_s: float | None = None) -> dict:
        del now_s
        present = [state for state in self._people.values() if state.present]
        speaker_status = self.speaker.status()
        return {
            "active": self.active,
            "mode": (
                "presence_compat"
                if self.auto_arrival_from_presence
                else "zone_events"
            ),
            "auto_arrival_from_presence": self.auto_arrival_from_presence,
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
        if not self.auto_arrival_from_presence:
            return
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
        if not self.auto_arrival_from_presence:
            self._pending_arrivals.clear()
            return
        if not self._pending_arrivals:
            return
        eligible_items = list(self._pending_arrivals.values())
        if self.greet_only_when_near:
            eligible_items = [
                item
                for item in eligible_items
                if (
                    (state := self._people.get(item.person_id)) is not None
                    and state.present
                    and state.near_camera
                )
            ]
        if not eligible_items:
            return
        earliest = min(item.created_s for item in eligible_items)
        required_delay = (
            self.arrival_group_window_s
            if self.greet_only_when_near
            else self.arrival_delay_s + self.arrival_group_window_s
        )
        if now_s - earliest < required_delay:
            return

        # Only greet people who stayed visible throughout the delay. Batch
        # people who arrived within the short group window; later arrivals
        # keep waiting for their own stable-presence interval.
        batch_limit = earliest + self.arrival_group_window_s
        arrivals: list[_PendingArrival] = []
        for person_id, item in list(self._pending_arrivals.items()):
            state = self._people.get(person_id)
            if state is None or not state.present:
                self._pending_arrivals.pop(person_id, None)
                continue
            if self.greet_only_when_near and not state.near_camera:
                continue
            if item.created_s > batch_limit:
                continue
            if now_s - state.last_seen_s > min(2.0, self.presence_timeout_s):
                self._pending_arrivals.pop(person_id, None)
                continue
            arrivals.append(item)
            self._pending_arrivals.pop(person_id, None)
        if not arrivals:
            return

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
            group_text = self._group_arrival_text(
                [item.display_name for item in arrivals]
            )
            if not self._is_cached(group_text):
                # Dynamic group combinations are not always pre-generated.
                # Reuse a cached, named sentence so cache-only mode responds
                # immediately instead of silently dropping the greeting.
                group_text = self._near_text(arrivals[0].display_name)
            accepted = self.speaker.say(
                group_text,
                key="arrival:group",
                cooldown_s=15.0,
                priority=PRIORITY_ARRIVAL,
                expires_s=10.0,
                fallback_text=group_text,
            )
            if accepted:
                for item in arrivals:
                    state = self._people.get(item.person_id)
                    if state:
                        state.last_arrival_spoken_s = now_s
                        state.mood = "happy"
                self._last_arrival_batch_s = now_s
                self._note_spoken(
                    "group_arrival", group_text, now_s, None, non_gesture=True
                )
            return

        item = arrivals[0]
        state = self._people.get(item.person_id)
        if state is None:
            return
        text = (
            self._return_text(state.display_name)
            if item.kind == "return"
            else (
                self._near_text(state.display_name)
                if self.greet_only_when_near
                else self._arrival_text(state.display_name)
            )
        )
        accepted = self.speaker.say(
            text,
            key=f"{item.kind}:{item.person_id}",
            cooldown_s=self.reentry_s,
            priority=PRIORITY_ARRIVAL,
            expires_s=12.0,
            fallback_text=text,
        )
        if accepted:
            state.last_arrival_spoken_s = now_s
            state.mood = "happy"
            self._last_arrival_batch_s = now_s
            self._note_spoken(item.kind, text, now_s, state, non_gesture=True)

    def _recent_wave_for_gid(self, global_id: int, now_s: float) -> bool:
        waved_at = self._recent_wave_gids.get(int(global_id), float("-inf"))
        return now_s - waved_at <= self.arrival_delay_s + self.wave_global_gap_s

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

    def _arrival_text(self, name: str) -> str:
        return self._pick_speakable_line(
            f"arrival-line:{name}",
            tuple(line.format(name=name) for line in ARRIVAL_TEMPLATES),
        )

    def _return_text(self, name: str) -> str:
        return self._pick_speakable_line(
            f"return-line:{name}",
            tuple(line.format(name=name) for line in RETURN_TEMPLATES),
        )

    def _near_text(self, name: str) -> str:
        return self._pick_speakable_line(
            f"near-line:{name}",
            tuple(line.format(name=name) for line in NEAR_GREETING_TEMPLATES),
        )

    @staticmethod
    def _group_arrival_text(names: list[str]) -> str:
        clean_names = [" ".join(str(name).split()) for name in names if name]
        if not clean_names:
            return GROUP_ARRIVAL
        if len(clean_names) == 1:
            return f"Chào {clean_names[0]} nha! Bé Xinh vui vì gặp bạn nè."
        # Naming one representative keeps a busy group greeting short and
        # lets the finite cache cover every possible group size/order.
        return (
            f"Chào {clean_names[0]} và mọi người nha! "
            "Bé Xinh vui vì gặp cả nhà nè."
        )

    def _pick_line(self, key: str, lines: tuple[str, ...]) -> str:
        if not lines:
            return ""
        index = self._line_counters.get(key, 0)
        self._line_counters[key] = index + 1
        return lines[index % len(lines)]

    def _is_cached(self, text: str) -> bool:
        """Treat speakers without cache introspection as fully available."""
        checker = getattr(self.speaker, "is_cached", None)
        if not callable(checker):
            return True
        try:
            return bool(checker(text))
        except Exception:
            return False

    def _pick_speakable_line(self, key: str, lines: tuple[str, ...]) -> str:
        """Rotate naturally while preferring lines ready for instant playback."""
        if not lines:
            return ""
        start = self._line_counters.get(key, 0)
        for offset in range(len(lines)):
            index = (start + offset) % len(lines)
            if self._is_cached(lines[index]):
                self._line_counters[key] = start + offset + 1
                return lines[index]
        return self._pick_line(key, lines)


def critical_prewarm_texts(people: Iterable[tuple[str, str]]) -> list[str]:
    """Small instant-response set: one proximity and one wave line/person."""
    texts: list[str] = []
    for _person_id, raw_name in people:
        name = " ".join(str(raw_name or "").split())
        if not name:
            continue
        texts.extend(
            [
                NEAR_GREETING_TEMPLATES[0].format(name=name),
                PERSONAL_WAVE_TEMPLATES[0].format(name=name),
            ]
        )
    return list(dict.fromkeys(texts))


def prewarm_texts(people: Iterable[tuple[str, str]]) -> list[str]:
    """Return every production sentence so live mode never synthesizes."""
    texts = [
        GENERIC_ARRIVAL,
        GROUP_ARRIVAL,
        *GENERIC_WAVE_LINES,
        GENERIC_APPROACH,
        GENERIC_STAND,
        *GENERIC_WATER_LINES,
        GROUP_WATER,
        *GENERIC_REST_LINES,
        *BANTER_LINES,
        *GROUP_BANTER_LINES,
    ]
    clean_people = [
        (person_id, " ".join(str(raw_name or "").split()))
        for person_id, raw_name in people
        if " ".join(str(raw_name or "").split())
    ]
    # Build the time-critical set for every person first. If a one-time cache
    # build is interrupted, nobody is left without a close/wave response.
    texts.extend(critical_prewarm_texts(clean_people))
    for _person_id, name in clean_people:
        texts.extend(
            [
                *(line.format(name=name) for line in ARRIVAL_TEMPLATES),
                *(line.format(name=name) for line in RETURN_TEMPLATES),
                *(line.format(name=name) for line in NEAR_GREETING_TEMPLATES),
                f"Ủa, {name} lại gần Bé Xinh hả?",
                f"Ơ, {name} đi đâu đó?",
                *(line.format(name=name) for line in PERSONAL_WAVE_TEMPLATES),
                f"Chào {name} và mọi người nha! Bé Xinh vui vì gặp cả nhà nè.",
                f"{name} ơi, uống miếng nước đi nha!",
                f"{name} ơi, ngồi lâu rồi. Duỗi người chút nha!",
            ]
        )
    return list(dict.fromkeys(texts))


__all__ = [
    "GENERIC_APPROACH",
    "GENERIC_ARRIVAL",
    "GENERIC_BANTER",
    "GENERIC_REST",
    "GENERIC_STAND",
    "GENERIC_WATER",
    "GENERIC_WAVE",
    "GROUP_ARRIVAL",
    "GROUP_BANTER",
    "GROUP_WATER",
    "HamyCompanion",
    "critical_prewarm_texts",
    "prewarm_texts",
]
