"""Image-zone R1/R2/R3 rules: center-in-polygon + stabilization."""
from camera_tracking.workstate.channel_status import (
    ChannelBusinessTracker,
    PersonBusinessState,
)
from camera_tracking.workstate.image_zones import (
    DEFAULT_R1,
    DEFAULT_R2,
    DEFAULT_R3,
    classify_channel_a,
    classify_channel_b,
    load_channel_zones,
    point_in_polygon,
)
from camera_tracking.workstate.room_fusion import (
    LABEL_AT_DOOR,
    LABEL_OUT_OF_DOOR,
    RoomPresenceAggregator,
)

INSIDE_R1 = (0.85, 0.90)
OUTSIDE_R1 = (0.30, 0.50)
INSIDE_R2 = (0.20, 0.85)
INSIDE_R3 = (0.60, 0.70)
OUTSIDE_B = (0.05, 0.20)


def _tracker_a(**kwargs) -> ChannelBusinessTracker:
    params = {"grace_s": 1.0, "dwell_s": 1.0, "away_grace_s": 3.0,
              "out_after_s": 20.0, "return_stable_s": 2.0,
              "zone_mode": "a_r1",
              "zone_r1": list(DEFAULT_R1),
              "zone_r2": list(DEFAULT_R2),
              "zone_r3": list(DEFAULT_R3)}
    params.update(kwargs)
    return ChannelBusinessTracker(channel="A", **params)


def _tracker_b(**kwargs) -> ChannelBusinessTracker:
    params = {"grace_s": 1.0, "dwell_s": 1.0, "away_grace_s": 3.0,
              "out_after_s": 20.0, "return_stable_s": 2.0,
              "zone_mode": "b_door",
              "zone_r1": list(DEFAULT_R1),
              "zone_r2": list(DEFAULT_R2),
              "zone_r3": list(DEFAULT_R3)}
    params.update(kwargs)
    return ChannelBusinessTracker(channel="B", **params)


def test_point_in_polygon_r1_r2_r3() -> None:
    assert point_in_polygon(INSIDE_R1, DEFAULT_R1)
    assert not point_in_polygon(OUTSIDE_R1, DEFAULT_R1)
    assert point_in_polygon(INSIDE_R2, DEFAULT_R2)
    assert not point_in_polygon(INSIDE_R2, DEFAULT_R3)
    assert point_in_polygon(INSIDE_R3, DEFAULT_R3)
    assert not point_in_polygon(INSIDE_R3, DEFAULT_R2)
    assert not point_in_polygon(OUTSIDE_B, DEFAULT_R2)
    assert not point_in_polygon(OUTSIDE_B, DEFAULT_R3)


def test_classify_helpers() -> None:
    assert classify_channel_a(INSIDE_R1, DEFAULT_R1) == "AWAY"
    assert classify_channel_a(OUTSIDE_R1, DEFAULT_R1) == "WORKING"
    assert classify_channel_b(INSIDE_R2, DEFAULT_R2, DEFAULT_R3) == "OUT_OF_DOOR"
    assert classify_channel_b(INSIDE_R3, DEFAULT_R2, DEFAULT_R3) == "AT_DOOR"
    assert classify_channel_b(OUTSIDE_B, DEFAULT_R2, DEFAULT_R3) == "AWAY"


def test_load_channel_zones_from_processed() -> None:
    zones = load_channel_zones("data/processed")
    assert set(zones) == {"R1", "R2", "R3"}
    assert all(len(poly) >= 3 for poly in zones.values())
    assert point_in_polygon(INSIDE_R1, zones["R1"])
    assert point_in_polygon(INSIDE_R2, zones["R2"])
    assert point_in_polygon(INSIDE_R3, zones["R3"])


def test_channel_a_r1_rule() -> None:
    tr = _tracker_a()
    assert tr.update(0.0, {1}, {1: OUTSIDE_R1})[1] is PersonBusinessState.UNKNOWN
    assert tr.update(1.0, {1}, {1: OUTSIDE_R1})[1] is PersonBusinessState.UNKNOWN
    assert tr.update(2.0, {1}, {1: OUTSIDE_R1})[1] is PersonBusinessState.WORKING


def test_channel_a_inside_r1_is_away() -> None:
    tr = _tracker_a()
    tr.update(0.0, {1}, {1: INSIDE_R1})
    tr.update(1.0, {1}, {1: INSIDE_R1})
    assert tr.update(2.0, {1}, {1: INSIDE_R1})[1] is PersonBusinessState.AWAY


def test_channel_a_flicker_keeps_working() -> None:
    tr = _tracker_a()
    tr.update(0.0, {1}, {1: OUTSIDE_R1})
    tr.update(1.0, {1}, {1: OUTSIDE_R1})
    tr.update(2.0, {1}, {1: OUTSIDE_R1})
    assert tr.update(2.5, {1}, {1: INSIDE_R1})[1] is PersonBusinessState.WORKING
    assert tr.update(3.0, {1}, {1: OUTSIDE_R1})[1] is PersonBusinessState.WORKING


def test_channel_b_r3_is_at_door() -> None:
    tr = _tracker_b()
    tr.update(0.0, {1}, {1: INSIDE_R3})
    tr.update(1.0, {1}, {1: INSIDE_R3})
    assert tr.update(2.0, {1}, {1: INSIDE_R3})[1] is PersonBusinessState.AT_DOOR


def test_channel_b_r2_is_out_of_door() -> None:
    tr = _tracker_b()
    tr.update(0.0, {1}, {1: INSIDE_R2})
    tr.update(1.0, {1}, {1: INSIDE_R2})
    assert tr.update(2.0, {1}, {1: INSIDE_R2})[1] is PersonBusinessState.OUT_OF_DOOR


def test_channel_b_default_is_away() -> None:
    tr = _tracker_b()
    tr.update(0.0, {1}, {1: OUTSIDE_B})
    tr.update(1.0, {1}, {1: OUTSIDE_B})
    assert tr.update(2.0, {1}, {1: OUTSIDE_B})[1] is PersonBusinessState.AWAY


def test_room_fusion_at_door() -> None:
    agg = RoomPresenceAggregator()
    out = agg.update(10.0, present_a=set(), present_b={1},
                     states_a={1: "AWAY"}, states_b={1: "AT_DOOR"},
                     names={})
    assert out[1].label == LABEL_AT_DOOR and out[1].in_room is True


def test_room_fusion_out_of_door() -> None:
    agg = RoomPresenceAggregator()
    out = agg.update(10.0, present_a=set(), present_b={1},
                     states_a={1: "AWAY"},
                     states_b={1: "OUT_OF_DOOR"}, names={})
    assert out[1].label == LABEL_OUT_OF_DOOR and out[1].in_room is False
    assert out[1].just_left_office is True
