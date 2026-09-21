import numpy as np

from camera_tracking.face.assignment import unique_face_assignments
from camera_tracking.face.gallery import EnrolledPerson


def _person(person_id: str) -> EnrolledPerson:
    return EnrolledPerson(
        person_id, person_id, np.array([1.0, 0.0], dtype=np.float32),
        employee_id=person_id,
    )


def test_same_top1_name_goes_to_higher_scoring_face() -> None:
    an, binh = _person("an"), _person("binh")
    result = unique_face_assignments(
        {1: [(an, 0.81), (binh, 0.79)], 2: [(an, 0.92), (binh, 0.40)]},
        threshold=0.70,
    )
    assert result[2].employee_id == "an"
    assert result[1].employee_id == "binh"
    assert result[1].rank == 2


def test_collision_loser_is_unknown_when_top2_is_below_threshold() -> None:
    an, binh = _person("an"), _person("binh")
    result = unique_face_assignments(
        {1: [(an, 0.81), (binh, 0.69)], 2: [(an, 0.92), (binh, 0.30)]},
        threshold=0.70,
    )
    assert result[2].employee_id == "an"
    assert 1 not in result


def test_assignment_is_unique_for_every_name_and_track() -> None:
    an, binh = _person("an"), _person("binh")
    result = unique_face_assignments(
        {1: [(an, 0.90), (binh, 0.80)], 2: [(an, 0.89), (binh, 0.88)]},
        threshold=0.70,
    )
    assert {candidate.employee_id for candidate in result.values()} == {"an", "binh"}
    assert result[1].employee_id == "an"
    assert result[2].employee_id == "binh"


def test_only_top_two_candidates_are_considered() -> None:
    an, binh, chi = _person("an"), _person("binh"), _person("chi")
    result = unique_face_assignments(
        {
            1: [(an, 0.95), (binh, 0.90), (chi, 0.89)],
            2: [(an, 0.96), (binh, 0.91)],
        },
        threshold=0.70,
    )
    assert result[2].employee_id == "an"
    assert result[1].employee_id == "binh"
    assert all(candidate.employee_id != "chi" for candidate in result.values())
