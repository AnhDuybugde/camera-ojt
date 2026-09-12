"""Reconcile unknown Global IDs with later face identifications.

Early in the day a person may be tracked as unknown (U-YYYYMMDD-###)
because their face was far/blurry/small. When the same human is later
face-identified as a known person, the stale unknown Global ID must be
merged into the known one instead of leaving a duplicate person behind.

Merge rule: cosine(unknown best face embedding, gallery embedding of the
identified person) >= threshold  ->  same human  ->  alias old gid to the
canonical (newly identified) gid. The caller then backfills person info
on the old row and marks it merged_into the canonical gid; the dashboard
hides merged rows. Attendance needs no merge: it is keyed by person_id
(1 tick/person/day) no matter how many gids the person had.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from camera_tracking.face.embeddings import cosine_similarity


@dataclass
class IdentityReconciler:
    """Collect unknown face embeddings, match them on later check-ins."""

    threshold: float = 0.5
    # owner_key (U-...) -> (best score, embedding)
    unknown_embs: dict[str, tuple[float, np.ndarray]] = field(
        default_factory=dict, init=False
    )

    def note_unknown(
        self,
        owner_key: str,
        embedding: np.ndarray | None,
        score: float = 0.0,
    ) -> None:
        """Remember the sharpest face embedding seen for an unknown owner."""
        if embedding is None:
            return
        prev = self.unknown_embs.get(owner_key)
        if prev is None or float(score) >= prev[0]:
            self.unknown_embs[owner_key] = (
                float(score),
                np.asarray(embedding, dtype=np.float32),
            )

    def forget(self, owner_key: str) -> None:
        self.unknown_embs.pop(owner_key, None)

    def find_merges(
        self,
        person_embedding: np.ndarray | None,
        unknown_of_gid: dict[int, str],
    ) -> dict[int, str]:
        """Return {old_gid: owner_key} whose face matches the known person."""
        if person_embedding is None:
            return {}
        merged: dict[int, str] = {}
        for gid, owner in unknown_of_gid.items():
            rec = self.unknown_embs.get(owner)
            if rec is None:
                continue
            if cosine_similarity(rec[1], person_embedding) >= self.threshold:
                merged[gid] = owner
        return merged


__all__ = ["IdentityReconciler"]
