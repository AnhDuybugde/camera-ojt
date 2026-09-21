"""Embedding math and image-quality checks."""
from __future__ import annotations

import cv2
import numpy as np


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize an empty/zero face embedding.")
    return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(normalize_embedding(left), normalize_embedding(right)))


def representative_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("No valid embeddings were captured.")
    normalized = np.stack([normalize_embedding(item) for item in embeddings])
    centroid = normalize_embedding(normalized.mean(axis=0))
    # Reject samples that strongly disagree with the centroid, then average again.
    scores = normalized @ centroid
    kept = normalized[scores >= max(0.25, float(np.median(scores) - 0.15))]
    return normalize_embedding(kept.mean(axis=0))


def blur_score(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def embedding_to_blob(embedding: np.ndarray) -> tuple[bytes, int]:
    vector = normalize_embedding(embedding).astype(np.float32)
    return vector.tobytes(), int(vector.size)


def embedding_from_blob(blob: bytes, dimension: int | None = None) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32).copy()
    if dimension and vector.size != dimension:
        raise ValueError(f"Invalid stored embedding: expected {dimension}, got {vector.size}.")
    return normalize_embedding(vector)

