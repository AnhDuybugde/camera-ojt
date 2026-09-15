"""So similarity 1 query embedding voi gallery enroll."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from camera_tracking.face.gallery import EnrolledPerson, FaceGallery


@dataclass(frozen=True, slots=True)
class MatchResult:
    person: EnrolledPerson | None
    score: float
    is_known: bool
    second_score: float = 0.0
    margin: float = 0.0


class FaceMatcher:
    def __init__(
        self,
        gallery: FaceGallery,
        threshold: float = 0.70,
        min_margin: float = 0.10,
    ) -> None:
        self.gallery = gallery
        self.threshold = threshold
        self.min_margin = max(0.0, min_margin)

    def match(self, query: np.ndarray | None) -> MatchResult:
        ranked = self.gallery.ranked_matches(query)
        if not ranked:
            return MatchResult(person=None, score=0.0, is_known=False)
        person, score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = float(score - second_score)
        known = score >= self.threshold and margin >= self.min_margin
        return MatchResult(
            person=person if known else None,
            score=float(score),
            is_known=known,
            second_score=float(second_score),
            margin=margin,
        )
