"""Tests for GlobalIdentityManager + ChannelBusinessTracker separation."""
from unittest import TestCase

import numpy as np

from camera_tracking.domain import BoundingBox, Track
from camera_tracking.tracking.global_identity import (
    GlobalIdentityConfig,
    GlobalIdentityManager,
    IdentityState,
)
from camera_tracking.workstate import ChannelBusinessTracker, PersonBusinessState


class MeanColorEmbedding:
    def extract(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        value = crop_bgr.mean(axis=(0, 1)).astype(np.float32)
        norm = float(np.linalg.norm(value))
        return value / norm if norm else None


def raw_track(track_id: int, bbox: BoundingBox) -> Track:
    return Track(track_id, bbox, 0.8, age=1, hits=2, confirmed=True)


def make_frame() -> np.ndarray:
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[40:160, 20:120] = (0, 0, 255)  # red person (left)
    frame[40:160, 260:360] = (255, 0, 0)  # blue person (right)
    return frame


def make_manager(**overrides) -> GlobalIdentityManager:
    cfg = GlobalIdentityConfig(**overrides)
    return GlobalIdentityManager(MeanColorEmbedding(), cfg)


class GlobalIdentityTest(TestCase):
    def test_tracklet_change_keeps_global_id(self) -> None:
        manager = make_manager()
        frame = make_frame()
        first = manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(100, BoundingBox(20, 40, 120, 160))], now_s=0.0,
        )
        # ByteTrack gives the same person a new tracklet id -> same Global ID.
        second = manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(999, BoundingBox(25, 42, 125, 162))], now_s=0.5,
        )
        self.assertEqual(first[0].track_id, 1)
        self.assertEqual(second[0].track_id, 1)
        self.assertEqual(first[0].local_track_id, 100)
        self.assertEqual(first[0].global_person_id, 1)
        self.assertIsNone(first[0].employee_id)

    def test_face_binding_is_metadata_separate_from_global_id(self) -> None:
        manager = make_manager()
        frame = make_frame()
        result = manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(10, BoundingBox(20, 40, 120, 160))], now_s=0.0,
        )

        manager.bind_employee(result[0].global_person_id, "employee_123")
        self.assertEqual(manager.employee_id_of(1), "employee_123")
        self.assertEqual(result[0].global_person_id, 1)
        refreshed = manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(10, BoundingBox(20, 40, 120, 160))], now_s=1.0,
        )
        self.assertEqual(refreshed[0].global_person_id, 1)
        self.assertEqual(refreshed[0].employee_id, "employee_123")

    def test_unbind_employee_clears_all_live_identity_metadata(self) -> None:
        manager = make_manager(min_appearance_similarity=1.1)
        frame = make_frame()
        first = manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(10, BoundingBox(20, 40, 120, 160))], now_s=0.0,
        )
        second = manager.update(
            channel="B", frame=frame,
            tracks=[raw_track(20, BoundingBox(260, 40, 360, 160))], now_s=0.1,
        )
        manager.bind_employee(first[0].global_person_id, "employee_123")
        manager.bind_employee(second[0].global_person_id, "employee_123")

        manager.unbind_employee("employee_123")

        self.assertIsNone(manager.employee_id_of(first[0].global_person_id))
        self.assertIsNone(manager.employee_id_of(second[0].global_person_id))

    def test_merge_identity_redirects_duplicate_tracklets(self) -> None:
        manager = make_manager()
        frame = make_frame()
        first = manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(10, BoundingBox(20, 40, 120, 160))], now_s=0.0,
        )
        second = manager.update(
            channel="B", frame=frame,
            tracks=[raw_track(20, BoundingBox(260, 40, 360, 160))], now_s=0.1,
        )
        manager.bind_employee(first[0].global_person_id, "employee_123")
        manager.bind_employee(second[0].global_person_id, "employee_123")

        assert manager.merge_identity(
            second[0].global_person_id, first[0].global_person_id
        )
        assert set(manager.identities) == {first[0].global_person_id}
        rebound = manager.update(
            channel="B", frame=frame,
            tracks=[raw_track(21, BoundingBox(260, 40, 360, 160))], now_s=0.2,
        )
        assert rebound[0].global_person_id == first[0].global_person_id

    def test_active_cross_channel_duplicates_need_face_confirmation(self) -> None:
        manager = make_manager(
            min_appearance_similarity=1.1,
            active_duplicate_similarity=0.8,
        )
        frame = make_frame()
        manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(10, BoundingBox(20, 40, 120, 160))], now_s=0.0,
        )
        manager.update(
            channel="B", frame=frame,
            tracks=[raw_track(20, BoundingBox(20, 40, 120, 160))], now_s=0.1,
        )
        assert manager.reconcile_active_duplicates() == {}
        assert set(manager.identities) == {1, 2}

        manager.bind_employee(1, "employee_123")
        manager.bind_employee(2, "employee_123")
        aliases = manager.reconcile_active_duplicates()
        assert aliases == {2: 1}
        assert set(manager.identities) == {1}

    def test_simultaneous_cross_channel_people_get_distinct_ids(self) -> None:
        manager = make_manager(min_appearance_similarity=0.0)
        frame = make_frame()
        first = manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(10, BoundingBox(20, 40, 120, 160))], now_s=10.0,
        )
        second = manager.update(
            channel="B", frame=frame,
            tracks=[raw_track(20, BoundingBox(20, 40, 120, 160))], now_s=10.0,
        )

        assert first[0].global_person_id == 1
        assert second[0].global_person_id == 2

    def test_active_same_camera_nested_duplicates_are_reconciled(self) -> None:
        manager = make_manager(
            min_appearance_similarity=1.1,
            active_duplicate_similarity=0.8,
        )
        frame = make_frame()
        manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(10, BoundingBox(20, 40, 120, 160))], now_s=0.0,
        )
        manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(11, BoundingBox(30, 50, 110, 150))], now_s=0.1,
        )
        aliases = manager.reconcile_active_duplicates()
        assert aliases == {2: 1}
        assert set(manager.identities) == {1}

    def test_long_disappearance_reconnects_same_global_id(self) -> None:
        manager = make_manager()
        frame = make_frame()
        manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(10, BoundingBox(20, 40, 120, 160))], now_s=0.0,
        )
        # Missing well past temp_lost_s (5s) into LONG_LOST, then reappears.
        for _ in range(3):
            manager.update(channel="A", frame=frame, tracks=[], now_s=30.0)
        self.assertEqual(manager.state_of(1), IdentityState.LONG_LOST)
        result = manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(77, BoundingBox(22, 41, 122, 161))], now_s=31.0,
        )
        self.assertEqual(result[0].track_id, 1)
        self.assertEqual(manager.state_of(1), IdentityState.ACTIVE)

    def test_cross_channel_reappearance_keeps_global_id(self) -> None:
        manager = make_manager()
        frame = make_frame()
        manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(10, BoundingBox(20, 40, 120, 160))], now_s=0.0,
        )
        # Same person (same red shirt) shows up on channel B, elsewhere in
        # the image -> same Global ID (appearance + time decide, not position).
        result = manager.update(
            channel="B", frame=frame,
            tracks=[raw_track(3, BoundingBox(30, 50, 110, 150))], now_s=8.0,
        )
        self.assertEqual(result[0].track_id, 1)

    def test_hungarian_matching_is_one_to_one_without_swap(self) -> None:
        manager = make_manager()
        frame = make_frame()
        first = manager.update(
            channel="A", frame=frame,
            tracks=[
                raw_track(10, BoundingBox(20, 40, 120, 160)),
                raw_track(20, BoundingBox(260, 40, 360, 160)),
            ],
            now_s=0.0,
        )
        # Both tracklets are replaced by new ids at slightly moved positions.
        second = manager.update(
            channel="A", frame=frame,
            tracks=[
                raw_track(30, BoundingBox(25, 40, 125, 160)),
                raw_track(40, BoundingBox(255, 40, 355, 160)),
            ],
            now_s=0.5,
        )
        self.assertEqual([t.track_id for t in first], [1, 2])
        self.assertEqual([t.track_id for t in second], [1, 2])

    def test_gating_rejects_different_far_person(self) -> None:
        manager = make_manager()
        frame = make_frame()
        manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(10, BoundingBox(20, 40, 120, 160))], now_s=0.0,
        )
        # Different appearance (blue) far from the red identity -> new Global ID.
        result = manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(11, BoundingBox(260, 40, 360, 160))], now_s=0.5,
        )
        self.assertEqual(len(result), 1)
        self.assertNotEqual(result[0].track_id, 1)

    def test_state_lifecycle_progresses(self) -> None:
        manager = make_manager()
        frame = make_frame()
        manager.update(
            channel="A", frame=frame,
            tracks=[raw_track(10, BoundingBox(20, 40, 120, 160))], now_s=0.0,
        )
        manager.update(channel="A", frame=frame, tracks=[], now_s=3.0)
        self.assertEqual(manager.state_of(1), IdentityState.TEMP_LOST)
        manager.update(channel="A", frame=frame, tracks=[], now_s=30.0)
        self.assertEqual(manager.state_of(1), IdentityState.LONG_LOST)
        manager.update(channel="A", frame=frame, tracks=[], now_s=250.0)
        self.assertEqual(manager.state_of(1), IdentityState.UNRESOLVED)
        # Even UNRESOLVED reconnects when the same person returns in time.
        result = manager.update(
            channel="B", frame=frame,
            tracks=[raw_track(5, BoundingBox(30, 50, 110, 150))], now_s=251.0,
        )
        self.assertEqual(result[0].track_id, 1)

    def test_gallery_holds_multiple_embeddings(self) -> None:
        manager = make_manager(gallery_size=4)
        frame = make_frame()
        for i in range(6):
            manager.update(
                channel="A", frame=frame,
                tracks=[raw_track(10, BoundingBox(20 + i, 40, 120 + i, 160))],
                now_s=float(i),
            )
        self.assertEqual(len(manager.identities[1].gallery), 4)


class BusinessSeparationTest(TestCase):
    def test_business_states_do_not_change_global_ids(self) -> None:
        manager = make_manager()
        business = ChannelBusinessTracker(
            channel="A", away_grace_s=10.0, out_after_s=60.0, return_stable_s=3.0
        )
        frame = make_frame()
        box = BoundingBox(20, 40, 120, 160)

        out = manager.update(channel="A", frame=frame, tracks=[raw_track(1, box)], now_s=0.0)
        states = business.update(0.0, {t.track_id for t in out})
        self.assertEqual(states[1], PersonBusinessState.WORKING)

        # Person leaves: business goes AWAY_TEMP then POSSIBLY_OUT...
        business.update(11.0, set())
        self.assertEqual(business.state_of(1), PersonBusinessState.AWAY_TEMP)
        business.update(61.0, set())
        self.assertEqual(business.state_of(1), PersonBusinessState.POSSIBLY_OUT)

        # ...but the Global ID reconnects when the same person returns.
        out = manager.update(
            channel="A", frame=frame, tracks=[raw_track(2, box)], now_s=70.0
        )
        self.assertEqual(out[0].track_id, 1)
        states = business.update(70.0, {1})
        self.assertEqual(states[1], PersonBusinessState.RETURNING)
        states = business.update(74.0, {1})
        self.assertEqual(states[1], PersonBusinessState.WORKING)
