"""Re-ID appearance embedding cho MVP.

MVP dùng histogram màu trang phục (fallback khi không thấy mặt).
Interface EmbeddingExtractor giữ nguyên để sau này thay bằng
ArcFace / InsightFace / OSNet mà không đổi engine.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingExtractor(Protocol):
    def extract(self, crop_bgr: np.ndarray | None) -> np.ndarray | None: ...


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).ravel()
    right = np.asarray(right, dtype=np.float32).ravel()
    if left.shape != right.shape or left.size == 0:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


class HistogramEmbedding:
    """Histogram HSV chuẩn hóa L2. Nhẹ, chạy CPU, đủ cho demo 1-2 người."""

    def __init__(self, h_bins: int = 16, s_bins: int = 16, v_bins: int = 8) -> None:
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.dim = h_bins * s_bins * v_bins

    def extract(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            import cv2
        except ImportError:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 8 or w < 8:
            return None
        # Lấy nửa thân dưới (áo/quần) ổn định hơn mặt khi camera sau lưng.
        torso = crop_bgr[h // 4 :, :, :]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [self.h_bins, self.s_bins, self.v_bins],
            [0, 180, 0, 256, 0, 256],
        )
        vec = hist.ravel().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return None
        return vec / norm

    def match(
        self,
        query: np.ndarray | None,
        candidates: list[np.ndarray | None],
        threshold: float,
    ) -> tuple[int | None, float]:
        """So 1 query với N candidates. Trả (best_idx|None, best_score)."""
        best_idx: int | None = None
        best_score = -1.0
        if query is None:
            return None, 0.0
        for idx, cand in enumerate(candidates):
            if cand is None:
                continue
            score = cosine_similarity(query, cand)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None or best_score < threshold:
            return None, max(best_score, 0.0)
        return best_idx, best_score
