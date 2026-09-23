from __future__ import annotations

from datetime import timedelta

from spatial.repository import SpatialRepository, local_now
from spatial.tracker import CentroidTracker
from spatial.zones import DEFAULT_ZONES, zone_at


def test_default_zones_cover_camera_width() -> None:
    assert zone_at(DEFAULT_ZONES, 0.10, 0.50).zone_id == "entrance"
    assert zone_at(DEFAULT_ZONES, 0.50, 0.50).zone_id == "center"
    assert zone_at(DEFAULT_ZONES, 0.90, 0.50).zone_id == "workspace"
    assert zone_at(DEFAULT_ZONES, 1.20, 0.50) is None


def test_tracker_keeps_id_and_retires_missing_track() -> None:
    tracker = CentroidTracker(max_distance=80, max_missed=1)
    first = tracker.update([(10, 10, 50, 100)])[0]
    second = tracker.update([(18, 12, 58, 102)])[0]
    assert first.track_id == second.track_id

    tracker.identify(second.track_id, "NV001", "Nguyễn Văn A")
    identified = tracker.snapshot(second.track_id)
    assert identified.employee_id == "NV001"

    tracker.update([])
    tracker.update([])
    retired = tracker.pop_retired()
    assert [item.track_id for item in retired] == [first.track_id]
    assert retired[0].employee_name == "Nguyễn Văn A"


def test_repository_isolated_density_movement_and_visit(tmp_path) -> None:
    repository = SpatialRepository(tmp_path / "spatial.db")
    now = local_now()
    repository.record_density("Camera 1", {"entrance": 2, "workspace": 1}, now)
    repository.record_transition(
        "Camera 1", 7, "NV001", "Nguyễn Văn A", "entrance", "workspace", now,
    )
    repository.record_visit(
        "Camera 1", 7, "NV001", "Nguyễn Văn A", "entrance",
        now - timedelta(seconds=45), now,
    )

    since = now - timedelta(minutes=1)
    assert len(repository.density_rows(since)) == 2
    assert repository.transition_rows(since)[0]["to_zone"] == "workspace"
    assert repository.visit_rows(since)[0]["dwell_seconds"] == 45
