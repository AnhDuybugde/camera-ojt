from __future__ import annotations

import json

import numpy as np

from camera_tracking.audio.zone_observer import ZoneEventObserver
from camera_tracking.domain import BoundingBox, Track


def _write_config(path, zones_a=None, zones_b=None) -> None:
    path.write_text(json.dumps({
        "version": 1,
        "coordinate_space": "normalized_image",
        "anchor": "bbox_bottom_center",
        "cameras": {
            "A": {"image_size": [1280, 720], "zones": zones_a or []},
            "B": {"image_size": [1280, 720], "zones": zones_b or []},
        },
    }), encoding="utf-8")


def _zone(zone_id, kind, points, **extra):
    return {
        "id": zone_id,
        "label": zone_id,
        "kind": kind,
        "points": points,
        "color": extra.get("color", "#22c55e"),
        "privacy": extra.get("privacy", "standard"),
        "enabled": extra.get("enabled", True),
    }


def _track(gid: int, foot_x: float, foot_y: float) -> Track:
    return Track(
        track_id=gid,
        bbox=BoundingBox(foot_x - 20, foot_y - 100, foot_x + 20, foot_y),
        confidence=0.9,
        age=3,
        hits=3,
        confirmed=True,
    )


def test_face_presence_alone_does_not_emit(tmp_path) -> None:
    path = tmp_path / "zones.json"
    _write_config(path, zones_a=[
        _zone("near", "be_xinh", [[0.6, 0.7], [1, 0.7], [1, 1], [0.6, 1]])
    ])
    observer = ZoneEventObserver(path, transition_dwell_s=0.25)
    track = _track(7, 1100, 650)

    assert observer.observe_tracks("A", [track], (720, 1280, 3), now_s=1.0) == []
    assert observer.observe_tracks("A", [track], (720, 1280, 3), now_s=1.3) == []


def test_be_xinh_zone_emits_once_after_dwell(tmp_path) -> None:
    path = tmp_path / "zones.json"
    _write_config(path, zones_a=[
        _zone("near", "be_xinh", [[0.6, 0.7], [1, 0.7], [1, 1], [0.6, 1]])
    ])
    observer = ZoneEventObserver(path, transition_dwell_s=0.2)
    track = _track(7, 1100, 650)
    kwargs = {
        "wall_time_s": 1_800_000_000.0,
        "person_ids": {7: "NV07"},
        "display_names": {7: "Ngọc"},
    }

    observer.observe_tracks("A", [track], (720, 1280, 3), now_s=1.0, **kwargs)
    observer.observe_tracks("A", [track], (720, 1280, 3), now_s=1.3, **kwargs)
    events = observer.observe_tracks("A", [track], (720, 1280, 3), now_s=2.2, **kwargs)
    assert len(events) == 1
    assert events[0]["type"] == "ZONE_DWELL"
    assert events[0]["zone"] == "be_xinh"
    assert events[0]["person_id"] == "NV07"
    assert observer.observe_tracks(
        "A", [track], (720, 1280, 3), now_s=3.0, **kwargs
    ) == []


def test_door_outside_to_inside_emits_enter(tmp_path) -> None:
    path = tmp_path / "zones.json"
    _write_config(path, zones_b=[
        _zone("outside", "door_outside", [[0, 0], [0.45, 0], [0.45, 1], [0, 1]]),
        _zone("inside", "door_inside", [[0.55, 0], [1, 0], [1, 1], [0.55, 1]]),
    ])
    observer = ZoneEventObserver(path, transition_dwell_s=0.1)
    outside = _track(9, 200, 500)
    inside = _track(9, 1000, 500)

    observer.observe_tracks("B", [outside], (720, 1280, 3), now_s=1.0)
    observer.observe_tracks("B", [outside], (720, 1280, 3), now_s=1.2)
    observer.observe_tracks("B", [inside], (720, 1280, 3), now_s=2.0)
    events = observer.observe_tracks(
        "B", [inside], (720, 1280, 3), now_s=2.2,
        wall_time_s=1_800_000_010.0,
    )

    assert len(events) == 1
    assert events[0]["type"] == "DOOR_ENTER"
    assert events[0]["direction"] == "outside_to_inside"


def test_door_inside_to_outside_emits_exit(tmp_path) -> None:
    path = tmp_path / "zones.json"
    _write_config(path, zones_b=[
        _zone("outside", "door_outside", [[0, 0], [0.45, 0], [0.45, 1], [0, 1]]),
        _zone("inside", "door_inside", [[0.55, 0], [1, 0], [1, 1], [0.55, 1]]),
    ])
    observer = ZoneEventObserver(path, transition_dwell_s=0.1)
    inside = _track(9, 1000, 500)
    outside = _track(9, 200, 500)

    observer.observe_tracks("B", [inside], (720, 1280, 3), now_s=1.0)
    observer.observe_tracks("B", [inside], (720, 1280, 3), now_s=1.2)
    observer.observe_tracks("B", [outside], (720, 1280, 3), now_s=2.0)
    events = observer.observe_tracks("B", [outside], (720, 1280, 3), now_s=2.2)

    assert events[0]["type"] == "DOOR_EXIT"
    assert events[0]["direction"] == "inside_to_outside"


def test_specific_small_zone_wins_overlap(tmp_path) -> None:
    path = tmp_path / "zones.json"
    _write_config(path, zones_a=[
        _zone("workspace", "workspace", [[0, 0], [1, 0], [1, 1], [0, 1]]),
        _zone("water", "water", [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]),
    ])
    observer = ZoneEventObserver(path)

    assert observer.zone_at("A", (0.5, 0.5)).kind == "water"


def test_event_expires_and_overlay_draws(tmp_path) -> None:
    path = tmp_path / "zones.json"
    _write_config(path, zones_a=[
        _zone("near", "be_xinh", [[0.6, 0.7], [1, 0.7], [1, 1], [0.6, 1]])
    ])
    observer = ZoneEventObserver(path, transition_dwell_s=0.0, event_ttl_s=2.0)
    track = _track(2, 1100, 650)
    observer.observe_tracks("A", [track], (720, 1280, 3), now_s=1.0)
    observer.observe_tracks("A", [track], (720, 1280, 3), now_s=2.0)
    assert observer.active_events(2.1)
    assert observer.active_events(5.0) == []
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert observer.draw_overlay(frame, "A").sum() > 0
