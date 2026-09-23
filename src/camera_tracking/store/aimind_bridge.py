"""Bridge camera-ojt -> ai-mind-attendance (phase 1: check-in/out only).

Producer (run_workstate*.py) pushes ``aimind_tick`` rows into the local
WriteQueue; the standalone consumer (scripts/bridge_aimind_attendance.py)
drains them through ai-mind-attendance's AttendanceService.record(), so
all schedule / WFH-OFF / break / cooldown / checkout rules stay in one
place and Google Sheets sync keeps working via PENDING rows.

Person identity: camera-ojt gallery person_id (e.g. "AnhDuy") is mapped
to ai-mind-attendance employee_id (e.g. "NV001") via
ai-mind-attendance/data/person_map.json. Unmapped or unknown persons are
skipped (acked + logged), never written: attendance.employee_id is a FK.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

AIMIND_TICK_KIND = "aimind_tick"

CHECK_IN = "CHECK_IN"
CHECK_OUT = "CHECK_OUT"

# record() actions that mean "handled, safe to ack".
_TERMINAL_ACTIONS = frozenset({
    "CHECK_IN", "CHECK_OUT", "WAITING", "COMPLETE", "COOLDOWN",
    "NO_SCHEDULE", "BREAK", "WFH", "OFF",
})

logger = logging.getLogger(__name__)


def build_checkin_payload(
    *,
    day: str,
    person_id: str,
    display_name: str,
    global_id: int,
    wall_iso: str,
    face_score: float,
) -> dict[str, Any]:
    return {
        "tick": CHECK_IN,
        "date": day,
        "person_id": person_id,
        "person_name": display_name,
        "global_id": int(global_id),
        "at": wall_iso,
        "face_score": float(face_score),
    }


def build_checkout_payload(
    *,
    day: str,
    person_id: str,
    display_name: str,
    global_id: int,
    wall_iso: str,
) -> dict[str, Any]:
    return {
        "tick": CHECK_OUT,
        "date": day,
        "person_id": person_id,
        "person_name": display_name,
        "global_id": int(global_id),
        "at": wall_iso,
    }


def load_person_map(path: str | Path) -> dict[str, str]:
    """Load gallery person_id -> ai-mind employee_id. Missing file -> {}."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if str(v).strip()}


def _parse_when(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def drain_once(queue: Any, service: Any, person_map: dict[str, str]) -> dict[str, int]:
    """Consume pending aimind_tick rows. Returns counters by outcome.

    Every row is acked unless the DB call itself raised (DB locked, ...):
    then it is left for the next poll. Unknown/unmapped persons and rule
    rejections (NO_SCHEDULE/BREAK/WFH/OFF/WAITING/...) are acked + counted,
    never retried: retrying cannot change the outcome.
    """
    stats: dict[str, int] = {
        "processed": 0, "checked_in": 0, "checked_out": 0,
        "waiting": 0, "skipped_unmapped": 0, "skipped_rule": 0, "errors": 0,
    }
    rows = (queue.peek_kind(AIMIND_TICK_KIND, 100)
            if callable(getattr(queue, "peek_kind", None)) else queue.peek(100))
    for row_id, kind, payload, _attempts in rows:
        if kind != AIMIND_TICK_KIND:
            continue
        stats["processed"] += 1
        person_id = str(payload.get("person_id") or "")
        employee_id = person_map.get(person_id, "")
        # New enrollments use the employee ID as camera person_id. Accept that
        # shared identifier without requiring a duplicate mapping file.
        if not employee_id:
            db = getattr(service, "db", None)
            get_employee = getattr(db, "get_employee", None)
            if callable(get_employee) and get_employee(person_id):
                employee_id = person_id
        if not employee_id:
            if payload.get("direction_confirmed"):
                from camera_tracking.store.event_log import append_event
                when = _parse_when(payload.get("at"))
                if when is None or when.tzinfo is None:
                    stats["errors"] += 1
                    continue
                with service.db.transaction() as conn:
                    append_event(conn, kind=payload["tick"], observed_at=when,
                        camera_id=payload.get("camera_id", ""),
                        session_id=payload.get("session_id", ""),
                        event_id=payload["event_id"], needs_review=True,
                        payload={"reason": "unconfirmed_identity", **payload.get("evidence", {})})
            logger.warning("aimind bridge: unmapped person %r, skip", person_id)
            stats["skipped_unmapped"] += 1
            if payload.get("direction_confirmed"):
                queue.ack(row_id)
            else:
                queue.quarantine(row_id, "unmapped_employee")
            continue
        when = _parse_when(payload.get("at"))
        if when is None or when.tzinfo is None:
            queue.quarantine(row_id, "invalid_observation_time")
            stats["errors"] += 1
            continue
        try:
            if payload.get("direction_confirmed"):
                result = service.record_direction(
                    employee_id, when, direction=payload["tick"],
                    event_id=payload["event_id"], camera_id=payload.get("camera_id", ""),
                    session_id=payload.get("session_id", ""), evidence=payload.get("evidence"))
            else:
                # Legacy observations have no directional evidence. They can
                # check in, but must never fabricate a checkout from a face hit.
                existing = service.db.fetch_one(
                    "SELECT id FROM attendance WHERE employee_id=? AND date=?",
                    (employee_id, when.date().isoformat()))
                if existing and payload.get("tick") == CHECK_IN:
                    stats["waiting"] += 1
                    queue.ack(row_id)
                    continue
                result = service.record(employee_id, when)
        except Exception:  # noqa: BLE001 - leave row for retry
            logger.exception("aimind bridge: record(%s) failed, retry later",
                             employee_id)
            stats["errors"] += 1
            continue
        action = getattr(result, "action", "")
        if action == "CHECK_IN":
            stats["checked_in"] += 1
        elif action == "CHECK_OUT":
            stats["checked_out"] += 1
        elif action in ("WAITING", "COMPLETE", "COOLDOWN", "DUPLICATE", "RETURN", "REVIEW"):
            stats["waiting"] += 1
        elif action in ("NO_SCHEDULE", "BREAK", "WFH", "OFF"):
            logger.info("aimind bridge: %s -> %s, ack", employee_id, action)
            stats["skipped_rule"] += 1
        else:  # ERROR (employee gone, ...) or future actions: ack + log.
            logger.warning("aimind bridge: %s -> %s, ack", employee_id, action)
            stats["skipped_rule"] += 1
        queue.ack(row_id)
    return stats


__all__ = [
    "AIMIND_TICK_KIND",
    "CHECK_IN",
    "CHECK_OUT",
    "build_checkin_payload",
    "build_checkout_payload",
    "drain_once",
    "load_person_map",
]
