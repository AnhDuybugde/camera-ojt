import numpy as np

from camera_tracking.face.gallery import EnrolledPerson, FaceGallery
from camera_tracking.face.matcher import FaceMatcher


def test_close_runner_up_remains_unknown() -> None:
    gallery = FaceGallery([
        EnrolledPerson("A", "A", np.array([1.0, 0.0], dtype=np.float32)),
        EnrolledPerson("B", "B", np.array([0.995, 0.1], dtype=np.float32)),
    ])
    match = FaceMatcher(gallery, threshold=0.70, min_margin=0.10).match(
        np.array([1.0, 0.0], dtype=np.float32)
    )
    assert match.is_known is False
    assert match.person is None


def test_clear_winner_can_match() -> None:
    gallery = FaceGallery([
        EnrolledPerson("A", "A", np.array([1.0, 0.0], dtype=np.float32)),
        EnrolledPerson("B", "B", np.array([0.0, 1.0], dtype=np.float32)),
    ])
    match = FaceMatcher(gallery, threshold=0.70, min_margin=0.10).match(
        np.array([1.0, 0.0], dtype=np.float32)
    )
    assert match.is_known is True
    assert match.person is gallery.people[0]
