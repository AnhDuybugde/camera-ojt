"""Test RoomPresenceAggregator: 5 labels, zone-direct door states."""
from camera_tracking.workstate.room_fusion import (
    LABEL_AT_DOOR,
    LABEL_AWAY,
    LABEL_OUT_OF_DOOR,
    LABEL_UNKNOWN,
    LABEL_WORKING,
    RoomPresenceAggregator,
)


def test_working_when_present_a() -> None:
    agg = RoomPresenceAggregator()
    out = agg.update(0.0, present_a={1}, present_b=set(),
                     states_a={1: "WORKING"}, states_b={}, names={1: "An"})
    assert out[1].label == LABEL_WORKING and out[1].in_room is True
    assert out[1].display_name == "An"


def test_away_without_door_evidence() -> None:
    agg = RoomPresenceAggregator()
    out = agg.update(0.0, present_a=set(), present_b=set(),
                     states_a={1: "AWAY"}, states_b={}, names={})
    assert out[1].label == LABEL_AWAY and out[1].in_room is True


def test_at_door_from_channel_b() -> None:
    agg = RoomPresenceAggregator()
    out = agg.update(0.0, present_a=set(), present_b={1},
                     states_a={1: "AWAY"}, states_b={1: "AT_DOOR"}, names={})
    assert out[1].label == LABEL_AT_DOOR and out[1].in_room is True
    assert out[1].just_left_office is False


def test_out_of_door_from_channel_b() -> None:
    agg = RoomPresenceAggregator()
    out = agg.update(0.0, present_a=set(), present_b={1},
                     states_a={1: "AWAY"}, states_b={1: "OUT_OF_DOOR"}, names={})
    assert out[1].label == LABEL_OUT_OF_DOOR and out[1].in_room is False
    assert out[1].just_left_office is True


def test_out_of_door_wins_over_simultaneous_a() -> None:
    """Overlap views: R2 outside-the-glass beats A WORKING."""
    agg = RoomPresenceAggregator()
    out = agg.update(0.0, present_a={1}, present_b={1},
                     states_a={1: "WORKING"}, states_b={1: "OUT_OF_DOOR"},
                     names={})
    assert out[1].label == LABEL_OUT_OF_DOOR and out[1].in_room is False


def test_long_absent_without_door_stays_away_when_disabled() -> None:
    """absent_fallback_s=0: absent without R2 evidence stays Away."""
    agg = RoomPresenceAggregator(absent_fallback_s=0)
    out = agg.update(0.0, present_a=set(), present_b=set(),
                     states_a={1: "AWAY"}, states_b={}, names={})
    assert out[1].label == LABEL_AWAY and out[1].in_room is True
    out = agg.update(5000.0, present_a=set(), present_b=set(),
                     states_a={1: "AWAY"}, states_b={}, names={})
    assert out[1].label == LABEL_AWAY and out[1].in_room is True


def test_absent_under_fallback_stays_away() -> None:
    agg = RoomPresenceAggregator(absent_fallback_s=900.0)
    agg.update(0.0, present_a={1}, present_b=set(),
               states_a={1: "WORKING"}, states_b={}, names={})
    out = agg.update(899.0, present_a=set(), present_b=set(),
                     states_a={1: "AWAY"}, states_b={}, names={})
    assert out[1].label == LABEL_AWAY and out[1].in_room is True
    assert out[1].just_left_office is False


def test_absent_past_fallback_becomes_out() -> None:
    agg = RoomPresenceAggregator(absent_fallback_s=900.0)
    agg.update(0.0, present_a={1}, present_b=set(),
               states_a={1: "WORKING"}, states_b={}, names={})
    out = agg.update(900.0, present_a=set(), present_b=set(),
                     states_a={1: "AWAY"}, states_b={}, names={})
    assert out[1].label == LABEL_OUT_OF_DOOR and out[1].in_room is False
    assert out[1].just_left_office is True
    # Edge fires only once.
    out = agg.update(901.0, present_a=set(), present_b=set(),
                     states_a={1: "AWAY"}, states_b={}, names={})
    assert out[1].label == LABEL_OUT_OF_DOOR
    assert out[1].just_left_office is False


def test_pruned_id_stays_away_not_unknown() -> None:
    """Channel pruned the ID (no states) but it was seen: Away, then Out."""
    agg = RoomPresenceAggregator(absent_fallback_s=900.0)
    agg.update(0.0, present_a={1}, present_b=set(),
               states_a={1: "WORKING"}, states_b={}, names={})
    out = agg.update(400.0, present_a=set(), present_b=set(),
                     states_a={}, states_b={}, names={})
    assert out[1].label == LABEL_AWAY and out[1].in_room is True
    out = agg.update(901.0, present_a=set(), present_b=set(),
                     states_a={}, states_b={}, names={})
    assert out[1].label == LABEL_OUT_OF_DOOR and out[1].in_room is False


def test_presence_resets_absent_timer() -> None:
    agg = RoomPresenceAggregator(absent_fallback_s=900.0)
    agg.update(0.0, present_a={1}, present_b=set(),
               states_a={1: "WORKING"}, states_b={}, names={})
    agg.update(800.0, present_a=set(), present_b=set(),
               states_a={1: "AWAY"}, states_b={}, names={})
    # Seen again at 850 (even in B): timer resets, no Out at 900.
    out = agg.update(850.0, present_a=set(), present_b={1},
                     states_a={1: "AWAY"}, states_b={1: "AWAY"}, names={})
    assert out[1].label == LABEL_AWAY
    out = agg.update(1700.0, present_a=set(), present_b=set(),
                     states_a={1: "AWAY"}, states_b={}, names={})
    assert out[1].label == LABEL_AWAY
    out = agg.update(1751.0, present_a=set(), present_b=set(),
                     states_a={1: "AWAY"}, states_b={}, names={})
    assert out[1].label == LABEL_OUT_OF_DOOR


def test_return_to_room() -> None:
    agg = RoomPresenceAggregator()
    agg.update(100.0, present_a=set(), present_b={1},
               states_a={1: "AWAY"}, states_b={1: "OUT_OF_DOOR"}, names={})
    out = agg.update(200.0, present_a={1}, present_b=set(),
                     states_a={1: "WORKING"}, states_b={}, names={1: "An"})
    assert out[1].label == LABEL_WORKING and out[1].just_returned is True


def test_unknown_when_nothing_seen() -> None:
    agg = RoomPresenceAggregator()
    out = agg.update(0.0, present_a=set(), present_b=set(),
                     states_a={1: "UNKNOWN"}, states_b={}, names={})
    assert out[1].label == LABEL_UNKNOWN and out[1].in_room is True
