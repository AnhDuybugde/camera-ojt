from __future__ import annotations

from camera_tracking.audio.events import (
    AudioEventKind,
    audio_event_stream_present,
    parse_audio_events,
)


def test_parse_semantic_event() -> None:
    events = parse_audio_events({"audio_events": [{
        "event_id": "e1",
        "type": "DOOR_ENTER",
        "timestamp": 1.0,
        "gid": 7,
        "person_id": "NV01",
        "name": "Ngọc",
    }]})
    assert len(events) == 1
    assert events[0].kind is AudioEventKind.DOOR_ENTER
    assert events[0].global_id == 7
    assert events[0].person_id == "NV01"
    assert events[0].display_name == "Ngọc"


def test_parse_generic_zone_dwell_and_wc_alias() -> None:
    events = parse_audio_events({"audio_events": [{
        "event_id": "e2",
        "type": "ZONE_DWELL",
        "zone": "wc",
        "timestamp": 2.0,
        "dwell_seconds": 1.2,
    }]})
    assert events[0].kind is AudioEventKind.RESTROOM
    assert events[0].zone == "restroom"
    assert events[0].dwell_seconds == 1.2


def test_invalid_rows_are_rejected() -> None:
    events = parse_audio_events({"audio_events": [
        {"type": "DOOR_ENTER", "timestamp": 1.0},
        {"event_id": "x", "type": "NOPE", "timestamp": 1.0},
        {"event_id": "y", "type": "WATER", "timestamp": 0},
        {"event_id": "z", "type": "WATER", "timestamp": 1.0, "confidence": 2},
    ]})
    assert events == []


def test_empty_event_list_still_switches_contract_mode() -> None:
    assert audio_event_stream_present({"audio_events": []})
    assert audio_event_stream_present({"audio": {"events": []}})
    assert not audio_event_stream_present({"people": []})


def test_identity_text_is_bounded_at_untrusted_boundary() -> None:
    event = parse_audio_events({"audio_events": [{
        "event_id": "bounded",
        "type": "WAVE",
        "timestamp": 1.0,
        "person_id": "P" * 500,
        "display_name": "N" * 500,
    }]})[0]
    assert len(event.person_id) == 100
    assert len(event.display_name) == 80
