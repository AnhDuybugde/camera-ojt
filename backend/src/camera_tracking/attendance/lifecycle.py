"""Auditable attendance lifecycle for camera-derived events.

The face recognizer supplies identity evidence; it never owns business state.
The lifecycle engine records the first verified entrance as CHECK_IN, confirmed
exits as TEMP_OUT + a checkout candidate, and a later entrance as RETURN.  The
latest confirmed exit is therefore the day's check-out candidate while a
return keeps the employee PRESENT.

This model is deliberately conservative: an UNKNOWN/low-confidence identity
must not create or modify attendance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AttendanceState(StrEnum):
    ABSENT = "ABSENT"
    PRESENT = "PRESENT"
    TEMP_OUT = "TEMP_OUT"
    CHECKED_OUT = "CHECKED_OUT"


@dataclass(slots=True, frozen=True)
class AttendanceLifecycleEvent:
    event: str
    day: str
    person_id: str
    display_name: str
    global_id: int
    at_iso: str
    channel: str
    confidence: float | None = None
    verification_method: str = "tracking+face"
    automatic: bool = True


@dataclass(slots=True)
class AttendanceLifecycleRecord:
    day: str
    person_id: str
    display_name: str
    state: AttendanceState = AttendanceState.ABSENT
    check_in_at: str | None = None
    check_out_at: str | None = None
    last_return_at: str | None = None
    last_event_at: str | None = None
    global_id: int | None = None
    confidence: float | None = None
    verification_method: str = "tracking+face"
    automatic: bool = True
    events: list[AttendanceLifecycleEvent] = field(default_factory=list)


class AttendanceLifecycleEngine:
    """Small deterministic state machine for first-in / last-out attendance."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], AttendanceLifecycleRecord] = {}

    def record_of(self, day: str, person_id: str) -> AttendanceLifecycleRecord | None:
        return self._records.get((str(day), str(person_id)))

    def records_for_day(self, day: str) -> list[AttendanceLifecycleRecord]:
        return sorted(
            (record for (record_day, _), record in self._records.items()
             if record_day == day),
            key=lambda record: record.person_id,
        )

    def preload(self, records: list[AttendanceLifecycleRecord]) -> None:
        for record in records:
            self._records[(record.day, record.person_id)] = record

    def observe_enter(
        self,
        *,
        day: str,
        person_id: str | None,
        display_name: str | None,
        global_id: int,
        at_iso: str,
        channel: str,
        confidence: float | None = None,
        verification_method: str = "tracking+face",
        automatic: bool = True,
    ) -> list[AttendanceLifecycleEvent]:
        if not person_id:
            return []
        record = self._ensure(
            day=day,
            person_id=person_id,
            display_name=display_name,
            global_id=global_id,
            confidence=confidence,
            verification_method=verification_method,
            automatic=automatic,
        )
        if record.check_in_at is None:
            record.check_in_at = at_iso
            record.check_out_at = None
            record.state = AttendanceState.PRESENT
            return [self._append(record, "CHECK_IN", at_iso, channel)]
        if record.state in (AttendanceState.TEMP_OUT, AttendanceState.CHECKED_OUT):
            record.state = AttendanceState.PRESENT
            record.last_return_at = at_iso
            # A later exit will replace this candidate. Keeping no stale
            # checkout while PRESENT avoids reports claiming the employee has
            # left when they have already returned.
            record.check_out_at = None
            return [self._append(record, "RETURN", at_iso, channel)]
        record.last_event_at = at_iso
        return []

    def observe_exit(
        self,
        *,
        day: str,
        person_id: str | None,
        display_name: str | None,
        global_id: int,
        at_iso: str,
        channel: str,
        confidence: float | None = None,
        verification_method: str = "tracking+face",
        automatic: bool = True,
        final: bool = False,
    ) -> list[AttendanceLifecycleEvent]:
        if not person_id:
            return []
        record = self._ensure(
            day=day,
            person_id=person_id,
            display_name=display_name,
            global_id=global_id,
            confidence=confidence,
            verification_method=verification_method,
            automatic=automatic,
        )
        # Never manufacture a check-in from an exit-only observation.
        if record.check_in_at is None:
            record.last_event_at = at_iso
            return []
        target_state = AttendanceState.CHECKED_OUT if final else AttendanceState.TEMP_OUT
        # Room fusion repeats a logical exit over several frames. Keep the
        # latest checkout candidate without emitting duplicate audit events.
        if record.state == target_state:
            if not record.check_out_at or at_iso > record.check_out_at:
                record.check_out_at = at_iso
            return []
        record.check_out_at = at_iso
        record.state = target_state
        event_name = "CHECK_OUT" if final else "TEMP_OUT"
        return [self._append(record, event_name, at_iso, channel)]

    def finalize_checkout(
        self, *, day: str, person_id: str, at_iso: str | None = None,
        channel: str = "SYSTEM"
    ) -> list[AttendanceLifecycleEvent]:
        record = self.record_of(day, person_id)
        if record is None or record.check_in_at is None:
            return []
        if record.state == AttendanceState.PRESENT:
            # Do not invent a physical exit if the employee is still present.
            return []
        timestamp = at_iso or record.check_out_at or record.last_event_at
        if not timestamp:
            return []
        record.check_out_at = timestamp
        if record.state == AttendanceState.CHECKED_OUT:
            return []
        record.state = AttendanceState.CHECKED_OUT
        return [self._append(record, "CHECK_OUT", timestamp, channel)]

    def _ensure(
        self, *, day: str, person_id: str, display_name: str | None,
        global_id: int, confidence: float | None, verification_method: str,
        automatic: bool,
    ) -> AttendanceLifecycleRecord:
        key = (str(day), str(person_id))
        record = self._records.get(key)
        if record is None:
            record = AttendanceLifecycleRecord(
                day=str(day), person_id=str(person_id),
                display_name=(display_name or person_id), global_id=int(global_id),
                confidence=confidence, verification_method=verification_method,
                automatic=bool(automatic),
            )
            self._records[key] = record
        else:
            record.display_name = display_name or record.display_name
            record.global_id = int(global_id)
            if confidence is not None:
                record.confidence = float(confidence)
            record.verification_method = verification_method
            record.automatic = bool(automatic)
        return record

    @staticmethod
    def _append(
        record: AttendanceLifecycleRecord, event: str, at_iso: str, channel: str
    ) -> AttendanceLifecycleEvent:
        item = AttendanceLifecycleEvent(
            event=event, day=record.day, person_id=record.person_id,
            display_name=record.display_name, global_id=int(record.global_id or 0),
            at_iso=at_iso, channel=channel, confidence=record.confidence,
            verification_method=record.verification_method,
            automatic=record.automatic,
        )
        record.last_event_at = at_iso
        record.events.append(item)
        return item
