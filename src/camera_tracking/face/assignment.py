"""Frame-level one-person/one-name assignment for face candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from camera_tracking.face.gallery import EnrolledPerson


@dataclass(frozen=True, slots=True)
class FaceCandidate:
    """One gallery candidate belonging to one detected person track."""

    person: EnrolledPerson
    score: float
    rank: int

    @property
    def employee_id(self) -> str:
        return self.person.employee_id or self.person.person_id


def unique_face_assignments(
    rankings: Mapping[int, Iterable[tuple[EnrolledPerson, float]]],
    *,
    threshold: float,
    limit: int = 2,
) -> dict[int, FaceCandidate]:
    """Assign at most one live track to each gallery identity.

    Every track contributes its own Top-N list.  Candidate edges are consumed
    from highest cosine score to lowest, so if two tracks want the same name,
    the higher-scoring track owns it.  The loser can then take its next free
    candidate, provided that candidate still meets the recognition threshold.
    """
    edges: list[tuple[float, int, int, EnrolledPerson]] = []
    for gid, ranked in rankings.items():
        seen_people: set[str] = set()
        for rank, (person, score) in enumerate(ranked, start=1):
            if rank > max(1, int(limit)):
                break
            employee_id = person.employee_id or person.person_id
            if employee_id in seen_people:
                continue
            seen_people.add(employee_id)
            numeric_score = float(score)
            if numeric_score >= float(threshold):
                edges.append((numeric_score, rank, int(gid), person))

    # Highest cosine owns a name.  Equal scores prefer Top 1, then stable GID.
    edges.sort(key=lambda item: (-item[0], item[1], item[2]))
    assigned_gids: set[int] = set()
    assigned_people: set[str] = set()
    output: dict[int, FaceCandidate] = {}
    for score, rank, gid, person in edges:
        employee_id = person.employee_id or person.person_id
        if gid in assigned_gids or employee_id in assigned_people:
            continue
        assigned_gids.add(gid)
        assigned_people.add(employee_id)
        output[gid] = FaceCandidate(person=person, score=score, rank=rank)
    return output


__all__ = ["FaceCandidate", "unique_face_assignments"]
