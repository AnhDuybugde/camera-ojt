"""Chong B cuop bbox/GID/ten cua A khi che khuat hoan toan."""
from unittest import TestCase

import numpy as np

from camera_tracking.domain import BoundingBox, Track
from camera_tracking.tracking.global_identity import (
    GlobalIdentityConfig,
    GlobalIdentityManager,
)


class MeanColorEmbedding:
    def extract(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        value = crop_bgr.mean(axis=(0, 1)).astype(np.float32)
        norm = float(np.linalg.norm(value))
        return value / norm if norm else None


def raw_track(track_id: int, bbox: BoundingBox) -> Track:
    return Track(track_id, bbox, 0.8, age=1, hits=2, confirmed=True)


def solid_frame(color: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[40:160, 20:120] = color  # vung A (trai)
    return frame


RED = (0, 0, 255)
BLUE = (255, 0, 0)
BOX_A = BoundingBox(20, 40, 120, 160)


def make_manager(**overrides) -> GlobalIdentityManager:
    cfg = GlobalIdentityConfig(
        gallery_refresh_steps=10,
        tentative_min_hits=1,
        **overrides,
    )
    return GlobalIdentityManager(MeanColorEmbedding(), cfg)


class OcclusionHijackTest(TestCase):
    def test_named_track_rejects_different_appearance_same_box(self) -> None:
        manager = make_manager()
        manager.update(
            channel="A", frame=solid_frame(RED),
            tracks=[raw_track(10, BOX_A)], now_s=0.0,
        )
        manager.bind_employee(1, "emp_a")
        # B mac ao xanh dung khop box cua A, cung raw track id (ByteTrack
        # giu id khi box gan nhau).
        result = manager.update(
            channel="A", frame=solid_frame(BLUE),
            tracks=[raw_track(10, BOX_A)], now_s=0.5,
        )
        # Khong duoc giu GID 1 cho body khac.
        self.assertNotEqual(result[0].track_id, 1)
        self.assertIsNone(result[0].employee_id)

    def test_gallery_not_polluted_by_hijack_attempt(self) -> None:
        manager = make_manager()
        manager.update(
            channel="A", frame=solid_frame(RED),
            tracks=[raw_track(10, BOX_A)], now_s=0.0,
        )
        before = len(manager.identities[1].gallery)
        manager.bind_employee(1, "emp_a")
        manager.update(
            channel="A", frame=solid_frame(BLUE),
            tracks=[raw_track(10, BOX_A)], now_s=0.5,
        )
        # Gallery cua A khong bi nhiem embedding xanh cua B.
        self.assertEqual(len(manager.identities[1].gallery), before)

    def test_face_locked_blocks_spatial_only_reconnect(self) -> None:
        manager = make_manager()
        manager.update(
            channel="A", frame=solid_frame(RED),
            tracks=[raw_track(10, BOX_A)], now_s=0.0,
        )
        manager.bind_employee(1, "emp_a")
        manager.set_face_anchor(1, "emp_a", 0.95, 0.0)
        # Mat embedding (ReID fail) ngay sau khi lock, B dung gan vi tri A.
        result = manager.update(
            channel="A", frame=solid_frame(BLUE),
            tracks=[raw_track(77, BoundingBox(22, 41, 122, 161))],
            now_s=1.0,
        )
        # ReID None? O day MeanColor van co embedding xanh -> khac gallery
        # nen bi gate. Test embedding None rieng bang _pair_score.
        self.assertNotEqual(result[0].track_id, 1)
        record = manager.identities[1]
        score = manager._pair_score(
            "A", BoundingBox(22, 41, 122, 161), (200, 400, 3),
            None, record, 1.0,
        )
        self.assertIsNone(score)

    def test_unlocked_spatial_reconnect_still_works_for_continuity(self) -> None:
        manager = make_manager()
        manager.update(
            channel="A", frame=solid_frame(RED),
            tracks=[raw_track(10, BOX_A)], now_s=0.0,
        )
        # Khong bind ten, khong lock: mat embedding + dung gan -> giu ID.
        record = manager.identities[1]
        score = manager._pair_score(
            "A", BoundingBox(22, 41, 122, 161), (200, 400, 3),
            None, record, 0.5,
        )
        self.assertIsNotNone(score)

    def test_unbind_clears_face_lock(self) -> None:
        manager = make_manager()
        manager.update(
            channel="A", frame=solid_frame(RED),
            tracks=[raw_track(10, BOX_A)], now_s=0.0,
        )
        manager.bind_employee(1, "emp_a")
        manager.set_face_anchor(1, "emp_a", 0.95, 0.0)
        self.assertTrue(manager._is_face_locked(manager.identities[1], 1.0))
        manager.unbind_employee(1)
        self.assertFalse(manager._is_face_locked(manager.identities[1], 1.0))
