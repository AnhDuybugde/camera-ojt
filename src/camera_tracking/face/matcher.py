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


class FaceMatcher:
    def __init__(
        self,
        gallery: FaceGallery,
        threshold: float = 0.5,
        min_margin: float = 0.0,
    ) -> None:
        self.gallery = gallery
        self.threshold = threshold
        self.min_margin = max(0.0, min_margin)

    def match(self, query: np.ndarray | None) -> MatchResult:
        person, score = self.gallery.best_match(
            query, self.threshold, self.min_margin
        )
        return MatchResult(person=person, score=float(score), is_known=person is not None)
