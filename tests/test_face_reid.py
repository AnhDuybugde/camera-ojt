"""FaceReIDEmbedding: nhan dang bang mat, quay lung tra None."""
import numpy as np

from camera_tracking.face.embeddings import FaceDetection
from camera_tracking.face.reid import FaceReIDEmbedding


def _vec(*vals: float) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float32)
    return arr / float(np.linalg.norm(arr))


class StubEmbedder:
    def __init__(self, dets):
        self.dets = dets

    def detect_embed(self, crop_bgr):
        return self.dets


def test_picks_best_face_over_threshold() -> None:
    small = FaceDetection((0, 0, 20, 20), 0.99, _vec(0.0, 1.0))
    big = FaceDetection((0, 0, 10, 10), 0.5, _vec(0.0, 1.0))
    good = FaceDetection((0, 0, 50, 60), 0.9, _vec(1.0, 0.0))
    reid = FaceReIDEmbedding(StubEmbedder([small, big, good]), min_face_px=40)
    out = reid.extract(np.zeros((100, 100, 3), dtype=np.uint8))
    assert out is not None
    # good (50x60, score 0.9) thang small (20px, score 0.99 bi loai size).
    np.testing.assert_allclose(out, _vec(1.0, 0.0), atol=1e-6)


def test_back_turned_returns_none() -> None:
    # Quay lung: khong thay mat nao -> None, manager khong ep match.
    reid = FaceReIDEmbedding(StubEmbedder([]), min_face_px=40)
    assert reid.extract(np.zeros((100, 100, 3), dtype=np.uint8)) is None
    assert reid.extract(None) is None


def test_all_faces_too_small_returns_none() -> None:
    tiny = FaceDetection((0, 0, 20, 20), 0.99, _vec(1.0, 0.0))
    reid = FaceReIDEmbedding(StubEmbedder([tiny]), min_face_px=40)
    assert reid.extract(np.zeros((100, 100, 3), dtype=np.uint8)) is None


def test_embedder_error_returns_none() -> None:
    class Broken:
        def detect_embed(self, crop_bgr):
            raise RuntimeError("no model")

    reid = FaceReIDEmbedding(Broken(), min_face_px=40)
    assert reid.extract(np.zeros((100, 100, 3), dtype=np.uint8)) is None


def test_back_turned_tracklets_never_merge() -> None:
    """2 nguoi quay lung (khong mat) o xa nhau -> 2 GID, khong gop bua."""
    from camera_tracking.domain import BoundingBox, Track
    from camera_tracking.tracking.global_identity import (
        GlobalIdentityConfig,
        GlobalIdentityManager,
    )

    class NoFace:
        def detect_embed(self, crop_bgr):
            return []

    reid = FaceReIDEmbedding(NoFace(), min_face_px=40)
    cfg = GlobalIdentityConfig(
        match_threshold=0.55, min_appearance_similarity=0.50,
        gallery_refresh_steps=1,
    )
    manager = GlobalIdentityManager(reid, cfg)
    frame = np.zeros((200, 400, 3), dtype=np.uint8)

    def raw(tid, box):
        return Track(tid, box, 0.8, age=1, hits=2, confirmed=True)

    first = manager.update(
        channel="A", frame=frame,
        tracks=[raw(10, BoundingBox(20, 40, 120, 160))], now_s=0.0)
    second = manager.update(
        channel="A", frame=frame,
        tracks=[raw(10, BoundingBox(20, 40, 120, 160)),
                raw(11, BoundingBox(260, 40, 360, 160))],
        now_s=0.5)
    gids = sorted(t.track_id for t in second)
    assert first[0].track_id == 1
    assert gids == [1, 2]
