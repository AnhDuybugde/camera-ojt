"""Enroll gallery tu thu muc `data/images/`."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from camera_tracking.face.embeddings import FaceEmbedder, cosine_similarity

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class EnrolledPerson:
    person_id: str  # stem file, vi du "LeHoAnhDuy"
    display_name: str  # ten hien thi tren overlay/dashboard
    embedding: np.ndarray | None  # None neu chua tinh duoc (van hien ten khi fallback)
    source_path: str = ""
    employee_id: str | None = None


@dataclass(slots=True)
class FaceGallery:
    people: list[EnrolledPerson] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.people)

    def best_match(
        self, query: np.ndarray | None, threshold: float
    ) -> tuple[EnrolledPerson | None, float]:
        """Tra (person | None, score). Khong dat nguong -> None."""
        if query is None or not self.people:
            return None, 0.0
        best: EnrolledPerson | None = None
        best_score = -1.0
        for person in self.people:
            if person.embedding is None:
                continue
            score = cosine_similarity(query, person.embedding)
            if score > best_score:
                best_score = score
                best = person
        if best is None or best_score < threshold:
            return None, max(best_score, 0.0)
        return best, float(best_score)


def _read_image(path: Path) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def load_gallery(
    gallery_dir: str | Path,
    embedder: FaceEmbedder | None,
    name_map: dict[str, str] | None = None,
    employee_map: dict[str, str] | None = None,
) -> FaceGallery:
    """Load one enrolled face per image and optional Employee ID mapping."""
    root = Path(gallery_dir)
    gallery = FaceGallery()
    if not root.is_dir():
        return gallery
    name_map = name_map or {}
    employee_map = employee_map or {}
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in _IMAGE_EXTS or not path.is_file():
            continue
        person_id = path.stem
        display_name = name_map.get(person_id, person_id)
        employee_id = employee_map.get(person_id)
        embedding: np.ndarray | None = None
        if embedder is not None:
            img = _read_image(path)
            if img is not None:
                try:
                    dets = embedder.detect_embed(img)
                except RuntimeError:
                    dets = []  # insightface chua cai -> enroll chay che do ten-only
                if dets:
                    embedding = np.asarray(dets[0].embedding, dtype=np.float32)
        gallery.people.append(
            EnrolledPerson(
                person_id=person_id,
                display_name=display_name,
                embedding=embedding,
                source_path=str(path),
                employee_id=employee_id,
            )
        )
    return gallery
