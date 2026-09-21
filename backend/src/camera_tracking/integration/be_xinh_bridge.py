"""Bridge the tracking status API to the event-driven Be Xinh companion.

The bridge deliberately stays outside ``run_workstate.py``.  The model team can
keep changing YOLO, ByteTrack, Re-ID and face recognition without creating a
large merge conflict with the audio implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Protocol
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class PersonSnapshot:
    """Small, validated projection of one person in ``/status.json``."""

    global_id: int
    person_id: str
    display_name: str
    in_room: bool
    label: str


class Companion(Protocol):
    """Subset of :class:`HamyCompanion` required by this adapter."""

    def identify(
        self,
        *,
        global_id: int,
        person_id: str,
        display_name: str,
        now_s: float,
    ) -> None: ...

    def observe_presence(
        self,
        *,
        person_id: str,
        display_name: str,
        now_s: float,
        stationary_since_s: float | None = None,
        global_id: int | None = None,
    ) -> None: ...

    def tick(self, now_s: float) -> None: ...

    def forget_global_ids(self, alive_global_ids: set[int]) -> None: ...


def parse_people(payload: dict[str, Any]) -> list[PersonSnapshot]:
    """Return only identified people with a usable Global ID.

    Unknown detections are intentionally ignored: speaking a guessed name is
    worse than remaining silent.
    """

    raw_people = payload.get("people", [])
    if not isinstance(raw_people, list):
        return []

    people: list[PersonSnapshot] = []
    for item in raw_people:
        if not isinstance(item, dict):
            continue
        try:
            global_id = int(item["gid"])
        except (KeyError, TypeError, ValueError):
            continue
        person_id = str(item.get("person_id") or "").strip()
        display_name = str(item.get("name") or "").strip()
        if not person_id or not display_name:
            continue
        people.append(
            PersonSnapshot(
                global_id=global_id,
                person_id=person_id,
                display_name=display_name,
                in_room=bool(item.get("in_room", True)),
                label=str(item.get("label") or ""),
            )
        )
    return people


class BackendStatusClient:
    """Read the backend status endpoint using only the Python standard library."""

    def __init__(self, status_url: str, timeout_s: float = 2.0) -> None:
        self.status_url = status_url.strip()
        self.timeout_s = max(0.1, float(timeout_s))

    def fetch(self) -> dict[str, Any]:
        request = Request(
            self.status_url,
            headers={"Accept": "application/json", "User-Agent": "be-xinh-bridge/1"},
        )
        with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
            if getattr(response, "status", 200) != 200:
                raise RuntimeError(f"tracking backend returned HTTP {response.status}")
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise TypeError("tracking backend status must be a JSON object")
        return payload


class BeXinhStatusBridge:
    """Translate model status snapshots into companion presence events."""

    def __init__(self, companion: Companion) -> None:
        self.companion = companion
        self._bindings: dict[int, tuple[str, str]] = {}

    def process(self, payload: dict[str, Any], now_s: float | None = None) -> int:
        now_s = time.monotonic() if now_s is None else float(now_s)
        alive_global_ids: set[int] = set()

        for person in parse_people(payload):
            if not person.in_room:
                continue
            alive_global_ids.add(person.global_id)
            binding = (person.person_id, person.display_name)
            if self._bindings.get(person.global_id) != binding:
                self.companion.identify(
                    global_id=person.global_id,
                    person_id=person.person_id,
                    display_name=person.display_name,
                    now_s=now_s,
                )
                self._bindings[person.global_id] = binding

            self.companion.observe_presence(
                person_id=person.person_id,
                display_name=person.display_name,
                now_s=now_s,
                global_id=person.global_id,
            )

        self.companion.forget_global_ids(alive_global_ids)
        self.companion.tick(now_s)

        retired = [gid for gid in self._bindings if gid not in alive_global_ids]
        for gid in retired:
            self._bindings.pop(gid, None)
        return len(alive_global_ids)
