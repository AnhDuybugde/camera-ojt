"""Face-based appearance cho Global Identity Manager.

Nhan dang theo GUONG MAT thay vi body (ao/quan): nguoi quay lung khong
con mat de so thi khong cap nhat gallery, khong ep match sai — ID giu
bang tracklet ngan han (ByteTrack) + spatial, cho den khi thay lai mat.
"""
from __future__ import annotations

import numpy as np

from camera_tracking.face.embeddings import FaceEmbedder


class FaceReIDEmbedding:
    """EmbeddingExtractor dung mat InsightFace thay vi crop body.

    Tra embedding L2-normalized cua mat tot nhat trong crop person,
    hoac None khi khong thay mat dat (quay lung / mat qua nho / mo).
    None nghia la "khong co bang chung ngoai hinh", manager se chi dung
    spatial + time + tracklet shortcut chu khong doan mo.
    """

    def __init__(
        self,
        embedder: FaceEmbedder,
        min_face_px: int = 40,
    ) -> None:
        self.embedder = embedder
        self.min_face_px = max(1, min_face_px)

    def extract(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            detections = self.embedder.detect_embed(crop_bgr)
        except RuntimeError:
            return None
        best = None
        for det in detections:
            width = det.bbox[2] - det.bbox[0]
            height = det.bbox[3] - det.bbox[1]
            if min(width, height) < self.min_face_px:
                continue
            if best is None or det.score > best.score:
                best = det
        if best is None:
            return None
        vector = np.asarray(best.embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else None


__all__ = ["FaceReIDEmbedding"]
