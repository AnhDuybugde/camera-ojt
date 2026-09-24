"""Policy router from semantic office events to the Bé Xinh speaker queue."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time
from typing import Callable, Iterable, Protocol

from camera_tracking.audio.dedupe import EventDedupe
from camera_tracking.audio.events import AudioEvent, AudioEventKind
from camera_tracking.audio.phrases import select_group_arrival_phrase, select_phrase
from camera_tracking.audio.policy import AudioPolicy


class SpeechSink(Protocol):
    @property
    def enabled(self) -> bool: ...

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


IdentityResolver = Callable[[int], tuple[str, str] | None]


@dataclass(frozen=True, slots=True)
class RouteResult:
    event_id: str
    accepted: bool
    reason: str
    kind: AudioEventKind
    line: str = ""


class AudioEventRouter:
    """Own privacy, priority, cooldown, de-duplication and queue expiry."""

    def __init__(
        self,
        speaker: SpeechSink,
        policy: AudioPolicy | None = None,
        *,
        turn_guard: Callable[[], bool] | None = None,
    ) -> None:
        self.speaker = speaker
        self.policy = policy or AudioPolicy.from_env()
        self._dedupe = EventDedupe(
            ttl_s=self.policy.dedupe_ttl_s,
            max_entries=self.policy.dedupe_max_entries,
        )
        self._last_by_subject_event: dict[tuple[AudioEventKind, str], float] = {}
        self._last_by_subject: dict[str, tuple[float, int]] = {}
        self._last_global_s = float("-inf")
        self._last_global_priority = 10_000
        self._last_event = "idle"
        self._last_line = ""
        self._counters: Counter[str] = Counter()
        self._turn_guard = turn_guard or (lambda: False)

    @property
    def enabled(self) -> bool:
        # Zone contract remains active while the physical speaker is muted.
        return self.policy.enabled

    def route_many(
        self,
        events: Iterable[AudioEvent],
        *,
        now_s: float | None = None,
        wall_time_s: float | None = None,
        resolve_identity: IdentityResolver | None = None,
    ) -> list[RouteResult]:
        now_s = time.monotonic() if now_s is None else float(now_s)
        wall_time_s = time.time() if wall_time_s is None else float(wall_time_s)
        items = list(events)
        if not self.policy.enabled:
            return [RouteResult(event.event_id, False, "disabled", event.kind) for event in items]

        prepared: list[AudioEvent] = []
        results: list[RouteResult] = []
        for event in items:
            # Validate dwell/age before de-duplication. A generic ZONE_DWELL
            # may keep the same ID while dwell_seconds grows to the threshold.
            reason = self._event_rejection_reason(event, wall_time_s)
            if reason:
                self._counters[reason] += 1
                results.append(RouteResult(event.event_id, False, reason, event.kind))
                continue
            if not self._dedupe.check_and_mark(event.event_id, now_s):
                self._counters["duplicate"] += 1
                results.append(RouteResult(event.event_id, False, "duplicate", event.kind))
                continue
            prepared.append(self._enrich_identity(event, resolve_identity))

        if self.policy.group_arrivals:
            arrivals = [event for event in prepared if event.kind is AudioEventKind.DOOR_ENTER]
            if len(arrivals) >= 2:
                group_result, consumed = self._route_group_arrival(arrivals, now_s)
                if group_result is not None:
                    results.append(group_result)
                consumed_ids = {event.event_id for event in consumed}
                prepared = [event for event in prepared if event.event_id not in consumed_ids]

        prepared.sort(key=lambda event: self.policy.for_kind(event.kind).priority)
        for event in prepared:
            results.append(self.route(event, now_s=now_s, already_deduped=True))
        return results

    def route(
        self,
        event: AudioEvent,
        *,
        now_s: float | None = None,
        wall_time_s: float | None = None,
        resolve_identity: IdentityResolver | None = None,
        already_deduped: bool = False,
    ) -> RouteResult:
        now_s = time.monotonic() if now_s is None else float(now_s)
        wall_time_s = time.time() if wall_time_s is None else float(wall_time_s)
        if not self.policy.enabled:
            return RouteResult(event.event_id, False, "disabled", event.kind)
        if not already_deduped:
            reason = self._event_rejection_reason(event, wall_time_s)
            if reason:
                self._counters[reason] += 1
                return RouteResult(event.event_id, False, reason, event.kind)
            if not self._dedupe.check_and_mark(event.event_id, now_s):
                self._counters["duplicate"] += 1
                return RouteResult(event.event_id, False, "duplicate", event.kind)
            event = self._enrich_identity(event, resolve_identity)

        policy = self.policy.for_kind(event.kind)
        subject = event.subject_key
        last_same = self._last_by_subject_event.get((event.kind, subject), float("-inf"))
        if now_s - last_same < policy.cooldown_s:
            self._counters["cooldown"] += 1
            return RouteResult(event.event_id, False, "cooldown", event.kind)

        recent = self._last_by_subject.get(subject)
        if recent is not None:
            last_s, last_priority = recent
            if now_s - last_s < self.policy.same_person_gap_s and policy.priority >= last_priority:
                self._counters["person_gap"] += 1
                return RouteResult(event.event_id, False, "person_gap", event.kind)
        if now_s - self._last_global_s < self.policy.global_gap_s and policy.priority >= self._last_global_priority:
            self._counters["global_gap"] += 1
            return RouteResult(event.event_id, False, "global_gap", event.kind)

        # Defence in depth: RESTROOM never receives a name even if a future
        # configuration accidentally enables names for that event policy.
        allow_name = policy.allow_name and event.kind is not AudioEventKind.RESTROOM
        phrase = select_phrase(
            event.kind,
            event_id=event.event_id,
            display_name=event.display_name,
            allow_name=allow_name,
        )
        accepted = self.speaker.say(
            phrase.text,
            key=f"zone:{event.kind.value}:{subject}",
            cooldown_s=policy.cooldown_s,
            priority=policy.priority,
            expires_s=policy.expires_s,
            fallback_text=phrase.fallback_text,
        )
        if not accepted:
            self._counters["speaker_rejected"] += 1
            return RouteResult(event.event_id, False, "speaker_rejected", event.kind)

        self._last_by_subject_event[(event.kind, subject)] = now_s
        self._last_by_subject[subject] = (now_s, policy.priority)
        self._last_global_s = now_s
        self._last_global_priority = policy.priority
        self._last_event = event.kind.value
        self._last_line = phrase.text
        self._counters["spoken"] += 1
        print(f"[Bé Xinh/Event {event.kind.value}] {phrase.text}")
        return RouteResult(event.event_id, True, "spoken", event.kind, phrase.text)

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "speaker_enabled": bool(getattr(self.speaker, "enabled", True)),
            "last_event": self._last_event,
            "last_line": self._last_line,
            "dedupe_size": len(self._dedupe),
            "counters": dict(self._counters),
        }

    def _route_group_arrival(
        self,
        arrivals: list[AudioEvent],
        now_s: float,
    ) -> tuple[RouteResult | None, list[AudioEvent]]:
        policy = self.policy.for_kind(AudioEventKind.DOOR_ENTER)
        eligible = [
            event for event in arrivals
            if now_s - self._last_by_subject_event.get(
                (event.kind, event.subject_key), float("-inf")
            ) >= policy.cooldown_s
        ]
        if len(eligible) < 2:
            return None, []
        if now_s - self._last_global_s < self.policy.global_gap_s and policy.priority >= self._last_global_priority:
            self._counters["global_gap"] += len(eligible)
            return (
                RouteResult("+".join(event.event_id for event in eligible), False, "global_gap", AudioEventKind.DOOR_ENTER),
                eligible,
            )

        phrase = select_group_arrival_phrase(event.event_id for event in eligible)
        accepted = self.speaker.say(
            phrase.text,
            key="zone:door_enter:group",
            cooldown_s=15.0,
            priority=policy.priority,
            expires_s=policy.expires_s,
            fallback_text=phrase.fallback_text,
        )
        event_id = "+".join(event.event_id for event in eligible)
        if not accepted:
            self._counters["speaker_rejected"] += len(eligible)
            return RouteResult(event_id, False, "speaker_rejected", AudioEventKind.DOOR_ENTER), eligible
        for event in eligible:
            self._last_by_subject_event[(event.kind, event.subject_key)] = now_s
            self._last_by_subject[event.subject_key] = (now_s, policy.priority)
        self._last_global_s = now_s
        self._last_global_priority = policy.priority
        self._last_event = "door_enter_group"
        self._last_line = phrase.text
        self._counters["spoken"] += 1
        print(f"[Bé Xinh/Event door_enter_group] {phrase.text}")
        return RouteResult(event_id, True, "spoken", AudioEventKind.DOOR_ENTER, phrase.text), eligible

    def _event_rejection_reason(self, event: AudioEvent, wall_time_s: float) -> str:
        if self._turn_guard():
            return "voice_turn"
        if event.timestamp >= 1_000_000_000:
            age = wall_time_s - event.timestamp
            if age > self.policy.max_event_age_s:
                return "stale"
            if age < -self.policy.future_tolerance_s:
                return "future"
        min_dwell = self.policy.for_kind(event.kind).generic_zone_min_dwell_s
        if min_dwell > 0 and event.is_generic_zone_event:
            if event.source_type != "zone_dwell":
                return "needs_dwell"
            if event.dwell_seconds is None or event.dwell_seconds < min_dwell:
                return "short_dwell"
        return ""

    @staticmethod
    def _enrich_identity(event: AudioEvent, resolve_identity: IdentityResolver | None) -> AudioEvent:
        if event.has_identity or event.global_id is None or resolve_identity is None:
            return event
        identity = resolve_identity(event.global_id)
        return event.with_identity(*identity) if identity is not None else event


__all__ = ["AudioEventRouter", "RouteResult"]
