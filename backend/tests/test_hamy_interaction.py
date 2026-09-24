from camera_tracking.audio.interaction import InteractionEngine
from camera_tracking.domain import BoundingBox, Track


def track(gid: int, box: tuple[float, float, float, float]) -> Track:
    return Track(
        track_id=gid,
        global_person_id=gid,
        bbox=BoundingBox(*box),
        confidence=0.9,
        age=10,
        hits=10,
        confirmed=True,
    )


def test_approach_detected_from_box_growth_and_motion():
    engine = InteractionEngine(approach_cooldown_s=30)
    shape = (1000, 1000, 3)
    assert not engine.observe_tracks("A", [track(1, (100, 200, 300, 500))], shape, 0.0)
    events = engine.observe_tracks("A", [track(1, (120, 130, 450, 600))], shape, 1.3)
    assert any(event.kind == "APPROACH" for event in events)


def test_stand_up_after_stable_period():
    engine = InteractionEngine(stand_cooldown_s=30)
    shape = (1000, 1000, 3)
    # Seated/stable box for several seconds.
    for sec in range(8):
        engine.observe_tracks("A", [track(1, (300, 400, 500, 700))], shape, float(sec))
    # Same foot line, top rises and bbox becomes much taller.
    events = engine.observe_tracks("A", [track(1, (300, 290, 500, 700))], shape, 8.0)
    assert any(event.kind == "STAND_UP" for event in events)


def test_proximity_uses_calibrated_distance_confirmation_and_hysteresis():
    engine = InteractionEngine(
        distance_reference_m=1.0,
        distance_reference_height_ratio=0.40,
        greeting_distance_m=0.50,
        distance_release_m=0.70,
        near_confirm_s=0.20,
    )
    shape = (1000, 1000, 3)

    # A height ratio of 0.80 maps to 0.50 m with this calibration.
    engine.observe_tracks("A", [track(1, (100, 100, 500, 900))], shape, 0.0)
    assert not engine.snapshot("A", 1).near_camera
    engine.observe_tracks("A", [track(1, (100, 100, 500, 900))], shape, 0.25)
    near = engine.snapshot("A", 1)
    assert near.near_camera
    assert near.estimated_distance_m == 0.5

    # Stay latched inside the 0.70 m release boundary, then release beyond it.
    engine.observe_tracks("A", [track(1, (100, 200, 500, 800))], shape, 0.50)
    assert engine.snapshot("A", 1).near_camera
    engine.observe_tracks("A", [track(1, (100, 250, 500, 750))], shape, 0.75)
    assert not engine.snapshot("A", 1).near_camera
