from camera_tracking.domain import BoundingBox, Track
from camera_tracking.gesture.focus import ForegroundSelector


def _track(gid: int, area: float) -> Track:
    return Track(gid, BoundingBox(0, 0, area, 1), 0.9, 3, 3, True)


def test_focus_uses_largest_and_ignores_small_jitter() -> None:
    focus = ForegroundSelector(switch_area_ratio=1.2, switch_hold_s=0.5)
    assert focus.update([_track(1, 100), _track(2, 90)], 0.0) == 1
    assert focus.update([_track(1, 100), _track(2, 115)], 1.0) == 1


def test_focus_switch_requires_margin_and_hold() -> None:
    focus = ForegroundSelector(switch_area_ratio=1.2, switch_hold_s=0.5)
    focus.update([_track(1, 100)], 0.0)
    assert focus.update([_track(1, 100), _track(2, 130)], 1.0) == 1
    assert focus.update([_track(1, 100), _track(2, 130)], 1.49) == 1
    assert focus.update([_track(1, 100), _track(2, 130)], 1.50) == 2


def test_focus_keeps_short_track_loss_and_follows_alias() -> None:
    focus = ForegroundSelector(lost_grace_s=1.0)
    focus.update([_track(7, 100)], 0.0)
    assert focus.update([_track(8, 200)], 0.5) == 7
    focus.remap({7: 3})
    assert focus.gid == 3
    assert focus.update([_track(3, 100)], 0.6) == 3
