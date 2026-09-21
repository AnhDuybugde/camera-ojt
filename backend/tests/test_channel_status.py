"""Test ChannelBusinessTracker: presence fallback + workstation ROI mode."""
from camera_tracking.workstate.channel_status import (
    ChannelBusinessTracker,
    PersonBusinessState,
)
from camera_tracking.workstate.workstation import WorkstationZone

DESK = WorkstationZone(
    name="desk1",
    core=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
    extended=[(-1.0, -1.0), (3.0, -1.0), (3.0, 3.0), (-1.0, 3.0)],
)
SEAT = (1.0, 1.0)
NEAR = (2.5, 1.0)
FAR = (5.0, 5.0)


def _zoned(**kwargs) -> ChannelBusinessTracker:
    params = {"workstations": [DESK], "grace_s": 1.0, "dwell_s": 1.0,
              "assign_dwell_s": 100.0, "hysteresis_m": 0.0,
              "away_grace_s": 3.0, "out_after_s": 20.0, "return_stable_s": 2.0}
    params.update(kwargs)
    return ChannelBusinessTracker(channel="A", **params)


def _commit(tr: ChannelBusinessTracker, now: float, pos) -> PersonBusinessState:
    return tr.update(now, {1}, {1: pos})[1]


SEAT_N = (0.5, 0.5)
FAR_N = (0.8, 0.5)
FAR2_N = (0.85, 0.55)


def _moved(**kwargs) -> ChannelBusinessTracker:
    params = {"grace_s": 1.0, "dwell_s": 1.0, "away_grace_s": 3.0,
              "out_after_s": 20.0, "return_stable_s": 2.0,
              "move_ratio": 0.15, "settle_ratio": 0.02}
    params.update(kwargs)
    return ChannelBusinessTracker(channel="A", **params)


def test_displaced_leave_seat_while_visible() -> None:
    tr = _moved()
    _commit(tr, 0.0, SEAT_N)
    _commit(tr, 1.0, SEAT_N)
    assert _commit(tr, 2.0, SEAT_N) is PersonBusinessState.WORKING
    assert _commit(tr, 3.0, FAR_N) is PersonBusinessState.WORKING  # grace
    assert _commit(tr, 4.0, FAR_N) is PersonBusinessState.WORKING  # dwell
    assert _commit(tr, 5.0, FAR_N) is PersonBusinessState.AWAY_TEMP
    # Roaming to another far spot: stays AWAY.
    assert _commit(tr, 6.0, FAR2_N) is PersonBusinessState.AWAY_TEMP


def test_displaced_still_elsewhere_is_away() -> None:
    """No motion at all, yet AWAY: position decides, not movement."""
    tr = _moved()
    _commit(tr, 0.0, SEAT_N)
    _commit(tr, 1.0, SEAT_N)
    _commit(tr, 2.0, SEAT_N)
    _commit(tr, 5.0, FAR_N)
    _commit(tr, 6.0, FAR_N)
    assert _commit(tr, 7.0, FAR_N) is PersonBusinessState.AWAY_TEMP
    assert _commit(tr, 8.0, FAR_N) is PersonBusinessState.AWAY_TEMP


def test_displaced_settle_new_spot_returns() -> None:
    tr = _moved()
    _commit(tr, 0.0, SEAT_N)
    _commit(tr, 1.0, SEAT_N)
    _commit(tr, 2.0, SEAT_N)
    _commit(tr, 5.0, FAR_N)
    _commit(tr, 6.0, FAR_N)
    _commit(tr, 7.0, FAR_N)  # adopts FAR_N as the new anchor here
    assert _commit(tr, 8.0, FAR_N) is PersonBusinessState.AWAY_TEMP
    assert _commit(tr, 9.0, FAR_N) is PersonBusinessState.AWAY_TEMP
    assert _commit(tr, 11.0, FAR_N) is PersonBusinessState.RETURNING
    assert _commit(tr, 13.0, FAR_N) is PersonBusinessState.WORKING


def test_displaced_flicker_keeps_working() -> None:
    tr = _moved()
    _commit(tr, 0.0, SEAT_N)
    _commit(tr, 1.0, SEAT_N)
    _commit(tr, 2.0, SEAT_N)
    assert _commit(tr, 2.5, FAR_N) is PersonBusinessState.WORKING
    assert _commit(tr, 3.0, SEAT_N) is PersonBusinessState.WORKING


def test_displaced_disabled_is_presence_only() -> None:
    tr = _moved(move_ratio=0.0)
    assert tr.update(0.0, {1}, {1: FAR_N})[1] is PersonBusinessState.WORKING
    assert tr.update(5.0, set())[1] is PersonBusinessState.AWAY_TEMP


def test_presence_fallback_without_workstations() -> None:
    tr = ChannelBusinessTracker(away_grace_s=3.0, out_after_s=20.0,
                                return_stable_s=2.0)
    assert tr.update(0.0, {1})[1] is PersonBusinessState.WORKING
    assert tr.update(2.0, set())[1] is PersonBusinessState.WORKING
    assert tr.update(4.0, set())[1] is PersonBusinessState.AWAY_TEMP
    assert tr.update(30.0, set())[1] is PersonBusinessState.POSSIBLY_OUT
    assert tr.update(31.0, {1})[1] is PersonBusinessState.RETURNING
    assert tr.update(34.0, {1})[1] is PersonBusinessState.WORKING


def test_zone_at_core_becomes_working() -> None:
    tr = _zoned()
    assert _commit(tr, 0.0, SEAT) is PersonBusinessState.UNKNOWN
    assert _commit(tr, 1.0, SEAT) is PersonBusinessState.UNKNOWN
    assert _commit(tr, 2.0, SEAT) is PersonBusinessState.WORKING


def test_zone_leave_seat_while_visible_is_away() -> None:
    """Standing still elsewhere still counts as AWAY (no motion term)."""
    tr = _zoned()
    _commit(tr, 0.0, SEAT)
    _commit(tr, 1.0, SEAT)
    _commit(tr, 2.0, SEAT)
    assert _commit(tr, 3.0, FAR) is PersonBusinessState.WORKING  # grace
    assert _commit(tr, 4.0, FAR) is PersonBusinessState.WORKING  # dwell
    assert _commit(tr, 5.0, FAR) is PersonBusinessState.AWAY_TEMP
    # Frozen in place far away: stays AWAY, never flips back by itself.
    assert _commit(tr, 6.0, FAR) is PersonBusinessState.AWAY_TEMP
    assert _commit(tr, 7.0, FAR) is PersonBusinessState.AWAY_TEMP


def test_zone_flicker_under_grace_keeps_state() -> None:
    tr = _zoned()
    _commit(tr, 0.0, SEAT)
    _commit(tr, 1.0, SEAT)
    _commit(tr, 2.0, SEAT)
    assert _commit(tr, 2.5, FAR) is PersonBusinessState.WORKING
    assert _commit(tr, 3.0, SEAT) is PersonBusinessState.WORKING
    assert _commit(tr, 5.0, SEAT) is PersonBusinessState.WORKING


def test_zone_near_seat() -> None:
    tr = _zoned()
    _commit(tr, 0.0, SEAT)
    _commit(tr, 1.0, SEAT)
    _commit(tr, 2.0, SEAT)
    assert _commit(tr, 5.0, NEAR) is PersonBusinessState.WORKING  # grace
    assert _commit(tr, 6.0, NEAR) is PersonBusinessState.WORKING  # dwell
    assert _commit(tr, 7.0, NEAR) is PersonBusinessState.NEAR_SEAT


def test_zone_return_cycle() -> None:
    tr = _zoned()
    _commit(tr, 0.0, SEAT)
    _commit(tr, 1.0, SEAT)
    _commit(tr, 2.0, SEAT)
    _commit(tr, 5.0, FAR)
    _commit(tr, 6.0, FAR)
    assert _commit(tr, 7.0, FAR) is PersonBusinessState.AWAY_TEMP
    assert _commit(tr, 8.0, SEAT) is PersonBusinessState.AWAY_TEMP  # grace
    assert _commit(tr, 9.0, SEAT) is PersonBusinessState.AWAY_TEMP  # dwell
    assert _commit(tr, 10.0, SEAT) is PersonBusinessState.RETURNING
    assert _commit(tr, 12.0, SEAT) is PersonBusinessState.WORKING


def test_zone_absence_still_times_out() -> None:
    tr = _zoned()
    _commit(tr, 0.0, SEAT)
    _commit(tr, 1.0, SEAT)
    _commit(tr, 2.0, SEAT)
    assert tr.update(3.0, set())[1] is PersonBusinessState.WORKING
    assert tr.update(6.0, set())[1] is PersonBusinessState.AWAY_TEMP
    assert tr.update(30.0, set())[1] is PersonBusinessState.POSSIBLY_OUT


def test_prune_forgets_dead_ids() -> None:
    tr = _zoned(prune_after_s=10.0)
    _commit(tr, 0.0, SEAT)
    _commit(tr, 1.0, SEAT)
    _commit(tr, 2.0, SEAT)
    tr.update(20.0, set())
    assert tr.state_of(1) is PersonBusinessState.UNKNOWN
    assert 1 not in tr._states
