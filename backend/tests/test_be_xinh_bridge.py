from __future__ import annotations

from camera_tracking.audio.events import AudioEventKind
from camera_tracking.integration.be_xinh_bridge import (
    BeXinhStatusBridge,
    parse_activity_events,
    parse_people,
)


class RecordingCompanion:
    def __init__(self) -> None:
        self.auto_arrival_from_presence = True
        self.arrival_modes: list[bool] = []
        self.identified: list[dict] = []
        self.observed: list[dict] = []
        self.alive: list[set[int]] = []
        self.ticks: list[float] = []
        self.waves: list[dict] = []
        self.motions: list[dict] = []

    def identify(self, **kwargs) -> None:
        self.identified.append(kwargs)

    def observe_presence(self, **kwargs) -> None:
        self.observed.append(kwargs)

    def forget_global_ids(self, gids: set[int]) -> None:
        self.alive.append(set(gids))

    def tick(self, now_s: float) -> None:
        self.ticks.append(now_s)

    def wave(self, **kwargs) -> bool:
        self.waves.append(kwargs)
        return True

    def motion_event(self, **kwargs) -> bool:
        self.motions.append(kwargs)
        return True

    def set_auto_arrival_from_presence(self, enabled: bool) -> None:
        self.auto_arrival_from_presence = bool(enabled)
        self.arrival_modes.append(bool(enabled))


def test_parse_people_rejects_unknown_and_invalid_rows() -> None:
    payload = {
        "people": [
            {
                "gid": 7,
                "person_id": "NV01",
                "name": "Ngọc",
                "in_room": True,
                "stationary_for_s": 42.5,
                "near_camera": True,
                "estimated_distance_m": 0.48,
            },
            {"gid": 8, "person_id": None, "name": "Unknown"},
            {"gid": "bad", "person_id": "NV02", "name": "An"},
        ]
    }

    people = parse_people(payload)

    assert [(p.global_id, p.person_id, p.display_name) for p in people] == [
        (7, "NV01", "Ngọc")
    ]
    assert people[0].stationary_for_s == 42.5
    assert people[0].near_camera is True
    assert people[0].estimated_distance_m == 0.48


def test_parse_people_ignores_lost_tracking_states() -> None:
    payload = {
        "people": [
            {
                "gid": 7,
                "person_id": "NV01",
                "name": "Ngọc",
                "in_room": True,
                "tracking_state": "ACTIVE",
            },
            {
                "gid": 8,
                "person_id": "NV02",
                "name": "An",
                "in_room": True,
                "tracking_state": "TEMP_LOST",
            },
            {
                "gid": 9,
                "person_id": "NV03",
                "name": "Bình",
                "in_room": True,
                "tracking_state": "LONG_LOST",
            },
        ]
    }

    assert [person.global_id for person in parse_people(payload)] == [7]


def test_parse_activity_events_validates_kind_and_numbers() -> None:
    events = parse_activity_events(
        {
            "gesture_events": [
                {"seq": 1, "kind": "open_palm", "gid": 7, "confidence": 1.4},
                {"seq": "bad", "kind": "OPEN_PALM", "gid": 8},
                {"seq": 3, "kind": "OTHER", "gid": 9},
            ]
        },
        "gesture_events",
        {"OPEN_PALM"},
    )

    assert len(events) == 1
    assert events[0].kind == "OPEN_PALM"
    assert events[0].confidence == 1.0


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


def test_bridge_forwards_gesture_and_motion_once_with_identity() -> None:
    companion = RecordingCompanion()
    bridge = BeXinhStatusBridge(companion)
    payload = {
        "people": [
            {
                "gid": 7,
                "person_id": "NV01",
                "name": "Ngọc",
                "in_room": True,
                "stationary_for_s": 20.0,
                "near_camera": True,
                "estimated_distance_m": 0.5,
            }
        ],
        "gesture_events": [
            {"seq": 11, "kind": "OPEN_PALM", "gid": 7, "channel": "A"}
        ],
        "motion_events": [
            {"seq": 12, "kind": "APPROACH", "gid": 7, "channel": "A"}
        ],
    }

    bridge.process(payload, now_s=100.0)
    bridge.process(payload, now_s=101.0)

    assert len(companion.waves) == 1
    assert companion.waves[0]["person_id"] == "NV01"
    assert len(companion.motions) == 1
    assert companion.motions[0]["kind"] == "APPROACH"
    assert companion.observed[0]["stationary_since_s"] == 80.0
    assert companion.observed[0]["near_camera"] is True
    assert companion.observed[0]["estimated_distance_m"] == 0.5


class RecordingRouter:
    enabled = True

    def __init__(self) -> None:
        self.batches = []
        self.resolved = []

    def route_many(self, events, *, now_s, wall_time_s, resolve_identity):
        self.batches.append(list(events))
        self.resolved.extend(
            resolve_identity(event.global_id)
            for event in events
            if event.global_id is not None
        )
        return []


def test_empty_audio_events_enables_zone_mode_and_disables_presence_arrival() -> None:
    companion = RecordingCompanion()
    router = RecordingRouter()
    bridge = BeXinhStatusBridge(companion, event_router=router)

    bridge.process(
        {"people": [{
            "gid": 7,
            "person_id": "NV01",
            "name": "Ngọc",
            "in_room": True,
            "near_camera": True,
            "estimated_distance_m": 0.48,
        }], "audio_events": []},
        now_s=10.0,
        wall_time_s=1_700_000_000.0,
    )

    assert bridge.zone_event_mode is True
    assert companion.auto_arrival_from_presence is False
    assert companion.observed[0]["near_camera"] is True
    assert companion.observed[0]["estimated_distance_m"] == 0.48


def test_zone_event_is_enriched_from_recent_identity() -> None:
    companion = RecordingCompanion()
    router = RecordingRouter()
    bridge = BeXinhStatusBridge(companion, event_router=router)

    bridge.process({
        "people": [{"gid": 7, "person_id": "NV01", "name": "Ngọc", "in_room": True}],
        "audio_events": [{
            "event_id": "door-1",
            "type": "DOOR_ENTER",
            "global_id": 7,
            "timestamp": 1_700_000_000.0,
        }],
    }, now_s=10.0, wall_time_s=1_700_000_000.0)

    assert router.batches[0][0].kind is AudioEventKind.DOOR_ENTER
    assert router.resolved == [("NV01", "Ngọc")]


def test_zone_mode_converts_legacy_open_palm_without_double_speech() -> None:
    companion = RecordingCompanion()
    router = RecordingRouter()
    bridge = BeXinhStatusBridge(companion, event_router=router)
    payload = {
        "people": [{"gid": 7, "person_id": "NV01", "name": "Ngọc", "in_room": True}],
        "audio_events": [],
        "gesture_events": [{"seq": 11, "kind": "OPEN_PALM", "gid": 7, "channel": "A"}],
        "motion_events": [{"seq": 12, "kind": "APPROACH", "gid": 7, "channel": "A"}],
    }

    bridge.process(payload, now_s=10.0, wall_time_s=1_700_000_000.0)
    bridge.process(payload, now_s=11.0, wall_time_s=1_700_000_001.0)

    routed = [event for batch in router.batches for event in batch]
    assert [event.kind for event in routed] == [AudioEventKind.WAVE]
    assert companion.waves == []
    assert companion.motions == []


def test_semantic_wave_suppresses_same_gid_legacy_wave() -> None:
    companion = RecordingCompanion()
    router = RecordingRouter()
    bridge = BeXinhStatusBridge(companion, event_router=router)
    bridge.process({
        "audio_events": [{
            "event_id": "wave-semantic",
            "type": "WAVE",
            "global_id": 7,
            "timestamp": 1_700_000_000.0,
        }],
        "gesture_events": [{"seq": 11, "kind": "OPEN_PALM", "gid": 7, "channel": "A"}],
    }, now_s=10.0, wall_time_s=1_700_000_000.0)

    assert len(router.batches[0]) == 1
    assert router.batches[0][0].event_id == "wave-semantic"


def test_bridge_restores_legacy_arrival_mode_when_event_contract_disappears() -> None:
    companion = RecordingCompanion()
    bridge = BeXinhStatusBridge(companion, event_router=RecordingRouter())
    bridge.process({"audio_events": []}, now_s=10.0, wall_time_s=100.0)
    bridge.process({"people": []}, now_s=11.0, wall_time_s=101.0)
    assert companion.arrival_modes[-2:] == [False, True]
