"""Test diem danh debounce: tick 1 lan/ngay, khong tick lap."""
import numpy as np

from camera_tracking.face.attendance import FaceAttendanceService
from camera_tracking.face.gallery import EnrolledPerson, FaceGallery
from camera_tracking.face.matcher import FaceMatcher


def _unit(dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.random(dim).astype(np.float32)
    return vec / float(np.linalg.norm(vec))


def test_debounce_ticks_once_per_day() -> None:
    svc = FaceAttendanceService(debounce_hits=3, window_s=5.0)
    day = "2026-09-12"
    assert svc.observe(day=day, global_id=1, person_id="An", display_name="An",
                       score=0.8, now_s=0.0, wall_time_iso=f"{day}T08:00:00") is None
    assert svc.observe(day=day, global_id=1, person_id="An", display_name="An",
                       score=0.82, now_s=1.0, wall_time_iso=f"{day}T08:00:01") is None
    ticked = svc.observe(day=day, global_id=1, person_id="An", display_name="An",
                         score=0.9, now_s=2.0, wall_time_iso=f"{day}T08:00:02")
    assert ticked is not None and ticked.display_name == "An"
    # Cung nguoi, cung ngay, du them hit nua cung khong tick lai.
    again = svc.observe(day=day, global_id=1, person_id="An", display_name="An",
                        score=0.95, now_s=3.0, wall_time_iso=f"{day}T08:00:03")
    assert again is None
    assert svc.is_ticked(day, "An")


def test_unknown_never_ticks() -> None:
    svc = FaceAttendanceService()
    assert svc.observe(day="2026-09-12", global_id=7, person_id=None,
                       display_name=None, score=0.1, now_s=0.0,
                       wall_time_iso="2026-09-12T08:00:00") is None


def test_window_expiry_resets_debounce() -> None:
    svc = FaceAttendanceService(debounce_hits=2, window_s=5.0)
    day = "2026-09-12"
    svc.observe(day=day, global_id=2, person_id="Bo", display_name="Bo",
                score=0.7, now_s=0.0, wall_time_iso=f"{day}T08:00:00")
    # Hit thu 2 cach 100s -> window het han -> chua du debounce.
    assert svc.observe(day=day, global_id=2, person_id="Bo", display_name="Bo",
                       score=0.7, now_s=100.0, wall_time_iso=f"{day}T08:01:40") is None


def test_matcher_threshold() -> None:
    query = _unit(seed=1)
    gallery = FaceGallery(people=[
        EnrolledPerson(person_id="An", display_name="An", embedding=query),
        EnrolledPerson(person_id="Bo", display_name="Bo", embedding=_unit(seed=99)),
    ])
    matcher = FaceMatcher(gallery, threshold=0.5)
    matched = matcher.match(query)
    assert matched.is_known and matched.person is not None and matched.person.person_id == "An"
    strict = FaceMatcher(gallery, threshold=1.01)
    assert strict.match(query).is_known is False
