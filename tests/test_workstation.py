"""Test workstation ROI observation + assignment (floor meters)."""
from camera_tracking.workstate.workstation import (
    WorkstationAssigner,
    WorkstationZone,
    ZoneObs,
    observe_zone,
)

DESK = WorkstationZone(
    name="desk1",
    core=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
    extended=[(-1.0, -1.0), (3.0, -1.0), (3.0, 3.0), (-1.0, 3.0)],
)


def test_observe_core_extended_outside() -> None:
    assert observe_zone((1.0, 1.0), DESK) is ZoneObs.AT_CORE
    assert observe_zone((2.5, 1.0), DESK) is ZoneObs.IN_EXTENDED
    assert observe_zone((5.0, 5.0), DESK) is ZoneObs.OUTSIDE
    assert observe_zone(None, DESK) is ZoneObs.UNKNOWN


def test_hysteresis_band_at_border() -> None:
    # Just outside the core edge: out without hysteresis...
    assert observe_zone((2.05, 1.0), DESK) is ZoneObs.IN_EXTENDED
    # ...but still inside while previously inside (0.3 m band).
    assert observe_zone((2.05, 1.0), DESK,
                        hysteresis_m=0.3, currently_inside=True) is ZoneObs.AT_CORE
    # Beyond the band even hysteresis gives up.
    assert observe_zone((2.5, 1.0), DESK,
                        hysteresis_m=0.3, currently_inside=True) is ZoneObs.IN_EXTENDED


def test_auto_assign_after_dwell() -> None:
    assigner = WorkstationAssigner(zones=[DESK], assign_dwell_s=5.0)
    zone, obs = assigner.observe(1, (1.0, 1.0), 0.0)
    assert obs is ZoneObs.AT_CORE and 1 not in assigner.assigned
    zone, obs = assigner.observe(1, (1.0, 1.0), 6.0)
    assert assigner.assigned[1] == "desk1"
    # Assigned: stickiness keeps reporting the same workstation.
    zone, obs = assigner.observe(1, (2.5, 1.0), 7.0)
    assert zone is not None and zone.name == "desk1"
    assert obs is ZoneObs.IN_EXTENDED


def test_person_map_override_and_transfer() -> None:
    assigner = WorkstationAssigner(
        zones=[DESK], assign_dwell_s=100.0, person_map={"An": "desk1"})
    zone, _ = assigner.observe(7, (5.0, 5.0), 0.0, person_id="An")
    assert zone is not None and assigner.assigned[7] == "desk1"
    assigner.transfer_assignment(7, 9)
    assert assigner.assigned.get(9) == "desk1"
    assert 7 not in assigner.assigned
    assigner.forget(9)
    assert 9 not in assigner.assigned
