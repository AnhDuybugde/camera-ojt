from __future__ import annotations

from camera_tracking.integration.be_xinh_bridge import BeXinhStatusBridge, parse_people


class RecordingCompanion:
    def __init__(self) -> None:
        self.identified: list[dict] = []
        self.observed: list[dict] = []
        self.alive: list[set[int]] = []
        self.ticks: list[float] = []

    def identify(self, **kwargs) -> None:
        self.identified.append(kwargs)

    def observe_presence(self, **kwargs) -> None:
        self.observed.append(kwargs)

    def forget_global_ids(self, gids: set[int]) -> None:
        self.alive.append(set(gids))

    def tick(self, now_s: float) -> None:
        self.ticks.append(now_s)


def test_parse_people_rejects_unknown_and_invalid_rows() -> None:
    payload = {
        "people": [
            {"gid": 7, "person_id": "NV01", "name": "Ngọc", "in_room": True},
            {"gid": 8, "person_id": None, "name": "Unknown"},
            {"gid": "bad", "person_id": "NV02", "name": "An"},
        ]
    }

    people = parse_people(payload)

    assert [(p.global_id, p.person_id, p.display_name) for p in people] == [
        (7, "NV01", "Ngọc")
    ]


def test_bridge_identifies_once_and_heartbeats_each_snapshot() -> None:
    companion = RecordingCompanion()
    bridge = BeXinhStatusBridge(companion)
    payload = {
        "people": [
            {"gid": 7, "person_id": "NV01", "name": "Ngọc", "in_room": True}
        ]
    }

    assert bridge.process(payload, now_s=10.0) == 1
    assert bridge.process(payload, now_s=11.0) == 1

    assert len(companion.identified) == 1
    assert len(companion.observed) == 2
    assert companion.alive == [{7}, {7}]
    assert companion.ticks == [10.0, 11.0]


def test_bridge_does_not_heartbeat_people_outside_room() -> None:
    companion = RecordingCompanion()
    bridge = BeXinhStatusBridge(companion)

    count = bridge.process(
        {"people": [{"gid": 9, "person_id": "NV09", "name": "An", "in_room": False}]},
        now_s=20.0,
    )

    assert count == 0
    assert companion.identified == []
    assert companion.observed == []
    assert companion.alive == [set()]
