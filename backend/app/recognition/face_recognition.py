"""In-memory, reloadable cosine face matcher."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from config import settings
from backend.app.database.database import Database
from face.embedding import embedding_from_blob, normalize_embedding

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Match:
    employee_id: str | None
    full_name: str
    department: str
    similarity: float

    @property
    def recognized(self) -> bool:
        return self.employee_id is not None


class FaceRecognizer:
    def __init__(self, db: Database, threshold: float | None = None) -> None:
        self.db = db
        self.threshold = settings.face_threshold if threshold is None else threshold
        self._lock = threading.RLock()
        self._matrix = np.empty((0, 0), dtype=np.float32)
        self._people: list[dict[str, str]] = []
        self.reload()

    def reload(self) -> int:
        vectors, people = [], []
        for row in self.db.list_embeddings():
            try:
                vectors.append(embedding_from_blob(row["face_embedding"], row["embedding_dim"]))
                people.append(row)
            except (ValueError, TypeError) as exc:
                logger.error("Skipping corrupt embedding for %s: %s", row["employee_id"], exc)
        with self._lock:
            self._matrix = np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)
            self._people = people
        return len(people)

    def match(self, embedding: np.ndarray) -> Match:
        query = normalize_embedding(embedding)
        with self._lock:
            if not self._people or self._matrix.shape[1] != query.size:
                return Match(None, "UNKNOWN", "", 0.0)
            scores = self._matrix @ query
            index = int(np.argmax(scores))
            score = float(scores[index])
            person = self._people[index]
        if score < self.threshold:
            return Match(None, "UNKNOWN", "", score)
        return Match(person["employee_id"], person["full_name"], person["department"], score)
