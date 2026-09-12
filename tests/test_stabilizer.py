"""Test StateStabilizer: raw -> stable -> committed separation."""
from camera_tracking.workstate.stabilizer import StateStabilizer, WorkstationState

AT = WorkstationState.AT_WORKSTATION
AWAY = WorkstationState.AWAY_FROM_WORKSTATION


def test_commit_needs_grace_plus_dwell() -> None:
    stab = StateStabilizer(grace_s=2.0, dwell_s=2.0)
    assert stab.update(0.0, AT) is WorkstationState.UNKNOWN
    assert stab.update(1.0, AT) is WorkstationState.UNKNOWN
    assert stab.update(2.0, AT) is WorkstationState.UNKNOWN  # stable now
    assert stab.update(4.0, AT) is AT  # dwelled -> committed


def test_flicker_under_grace_ignored() -> None:
    stab = StateStabilizer(grace_s=3.0, dwell_s=1.0)
    stab.update(0.0, AT)
    stab.update(4.0, AT)
    assert stab.update(5.0, AT) is AT
    assert stab.update(6.0, AWAY) is AT  # 1s excursion
    assert stab.update(7.0, AT) is AT
    assert stab.update(10.0, AT) is AT


def test_sustained_leave_commits_away() -> None:
    stab = StateStabilizer(grace_s=2.0, dwell_s=2.0)
    stab.update(0.0, AT)
    stab.update(4.0, AT)
    assert stab.update(6.0, AT) is AT
    assert stab.update(7.0, AWAY) is AT
    assert stab.update(9.0, AWAY) is AT  # stable flipped, dwell pending
    assert stab.update(11.0, AWAY) is AWAY


def test_urgency_only_shortens_within_bounds() -> None:
    fast = StateStabilizer(grace_s=4.0, dwell_s=4.0, motion_influence=1.0)
    slow = StateStabilizer(grace_s=4.0, dwell_s=4.0, motion_influence=0.0)
    # urgency=1 halves timings: commit at t=4 instead of t=8.
    fast.update(0.0, AT, urgency=1.0)
    assert fast.update(2.0, AT, urgency=1.0) is WorkstationState.UNKNOWN
    assert fast.update(4.0, AT, urgency=1.0) is AT
    slow.update(0.0, AT)
    assert slow.update(4.0, AT) is WorkstationState.UNKNOWN
    assert slow.update(8.0, AT) is AT
