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
