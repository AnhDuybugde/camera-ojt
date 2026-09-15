"""Test RoomPresenceAggregator: English label mapping + in_room."""
from camera_tracking.workstate.room_fusion import (
    LABEL_AWAY_SEAT,
    LABEL_NEAR_SEAT,
    LABEL_OUT_OFFICE,
    LABEL_WORKING,
    RoomPresenceAggregator,
)


def test_working_when_present_a() -> None:
    agg = RoomPresenceAggregator()
    out = agg.update(0.0, present_a={1}, present_b=set(),
                     states_a={1: "WORKING"}, states_b={}, names={1: "An"})
    assert out[1].label == LABEL_WORKING and out[1].in_room is True
    assert out[1].display_name == "An"


def test_away_seat_without_door_evidence() -> None:
    agg = RoomPresenceAggregator()
    out = agg.update(0.0, present_a=set(), present_b=set(),
                     states_a={1: "AWAY_TEMP"}, states_b={}, names={})
    assert out[1].label == LABEL_AWAY_SEAT and out[1].in_room is True


def test_near_seat_without_door_evidence() -> None:
    agg = RoomPresenceAggregator()
    out = agg.update(0.0, present_a={1}, present_b=set(),
                     states_a={1: "NEAR_SEAT"}, states_b={}, names={})
    assert out[1].label == LABEL_NEAR_SEAT and out[1].in_room is True


def test_leave_office_confirmed_by_channel_b() -> None:
    agg = RoomPresenceAggregator(leave_confirm_window_s=300.0)
    out = agg.update(100.0, present_a=set(), present_b={1},
                     states_a={1: "POSSIBLY_OUT"}, states_b={1: "WORKING"}, names={})
    assert out[1].label == LABEL_OUT_OFFICE and out[1].in_room is False
    assert out[1].just_left_office is True


def test_long_absent_without_door_stays_away_seat() -> None:
    agg = RoomPresenceAggregator()
    out = agg.update(0.0, present_a=set(), present_b=set(),
                     states_a={1: "POSSIBLY_OUT"}, states_b={}, names={})
    assert out[1].label == LABEL_AWAY_SEAT and out[1].in_room is True


def test_return_to_room() -> None:
    agg = RoomPresenceAggregator(leave_confirm_window_s=300.0)
    agg.update(100.0, present_a=set(), present_b={1},
               states_a={1: "POSSIBLY_OUT"}, states_b={1: "WORKING"}, names={})
    out = agg.update(200.0, present_a={1}, present_b=set(),
                     states_a={1: "WORKING"}, states_b={}, names={1: "An"})
    assert out[1].label == LABEL_WORKING and out[1].just_returned is True


def test_absent_fallback_marks_out_without_door() -> None:
    agg = RoomPresenceAggregator(absent_fallback_s=900.0)
    agg.update(0.0, present_a=set(), present_b=set(),
               states_a={1: "POSSIBLY_OUT"}, states_b={}, names={})
    out = agg.update(901.0, present_a=set(), present_b=set(),
                     states_a={1: "POSSIBLY_OUT"}, states_b={}, names={})
    assert out[1].label == LABEL_OUT_OFFICE and out[1].in_room is False


def test_absent_fallback_disabled_stays_away() -> None:
    agg = RoomPresenceAggregator(absent_fallback_s=0)
    agg.update(0.0, present_a=set(), present_b=set(),
               states_a={1: "POSSIBLY_OUT"}, states_b={}, names={})
    out = agg.update(5000.0, present_a=set(), present_b=set(),
                     states_a={1: "POSSIBLY_OUT"}, states_b={}, names={})
    assert out[1].label == LABEL_AWAY_SEAT and out[1].in_room is True
