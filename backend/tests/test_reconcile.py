"""Test IdentityReconciler: unknown GID merges into known on check-in."""
import numpy as np

from camera_tracking.workstate.reconcile import IdentityReconciler


def _vec(*vals: float) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float32)
    return arr / float(np.linalg.norm(arr))


def test_merge_matching_unknown() -> None:
    rec = IdentityReconciler(threshold=0.5)
    gallery_emb = _vec(1.0, 0.0, 0.0, 0.0)
    # Same human, slightly noisy capture while unknown.
    rec.note_unknown("U-20260912-001", _vec(0.95, 0.1, 0.0, 0.0), score=0.8)
    # Someone else entirely.
    rec.note_unknown("U-20260912-002", _vec(0.0, 0.0, 0.0, 1.0), score=0.7)
    merges = rec.find_merges(gallery_emb, {5: "U-20260912-001", 7: "U-20260912-002"})
    assert merges == {5: "U-20260912-001"}


def test_no_merge_without_embedding() -> None:
    rec = IdentityReconciler(threshold=0.5)
    rec.note_unknown("U-20260912-001", None, score=0.9)
    assert rec.find_merges(_vec(1.0, 0.0), {5: "U-20260912-001"}) == {}
    assert rec.find_merges(None, {5: "U-20260912-001"}) == {}


def test_best_embedding_kept() -> None:
    rec = IdentityReconciler(threshold=0.99)
    rec.note_unknown("U-1", _vec(0.0, 1.0), score=0.5)
    rec.note_unknown("U-1", _vec(1.0, 0.0), score=0.9)
    assert rec.find_merges(_vec(1.0, 0.0), {3: "U-1"}) == {3: "U-1"}
    rec.forget("U-1")
    assert rec.find_merges(_vec(1.0, 0.0), {3: "U-1"}) == {}
