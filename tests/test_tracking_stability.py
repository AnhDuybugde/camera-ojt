from unittest import TestCase

import numpy as np

from camera_tracking.domain import BoundingBox, Track
from camera_tracking.tracking import PersistentIdentityTracker, StablePersonCount


class MeanColorEmbedding:
    def extract(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        value = crop_bgr.mean(axis=(0, 1)).astype(np.float32)
        norm = float(np.linalg.norm(value))
        return value / norm if norm else None


def raw_track(track_id: int, bbox: BoundingBox) -> Track:
    return Track(track_id, bbox, 0.8, age=1, hits=2, confirmed=True)


class StablePersonCountTest(TestCase):
    def test_single_frame_miss_does_not_change_count(self) -> None:
        counter = StablePersonCount(rise_frames=2, fall_frames=3)
        counter.update(5)
        self.assertEqual(counter.update(5), 5)

        self.assertEqual(counter.update(4), 5)
        self.assertEqual(counter.update(5), 5)

    def test_sustained_change_updates_count(self) -> None:
        counter = StablePersonCount(rise_frames=2, fall_frames=3)

        self.assertEqual(counter.update(5), 0)
        self.assertEqual(counter.update(5), 5)
        self.assertEqual(counter.update(4), 5)
        self.assertEqual(counter.update(4), 5)
        self.assertEqual(counter.update(4), 4)


class PersistentIdentityTrackerTest(TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((200, 400, 3), dtype=np.uint8)
        self.frame[40:160, 20:120] = (0, 0, 255)
        self.frame[40:160, 260:360] = (255, 0, 0)

    def test_raw_id_change_keeps_stable_identity(self) -> None:
        tracker = PersistentIdentityTracker(MeanColorEmbedding())
        first = tracker.update(
            self.frame,
            [raw_track(100, BoundingBox(20, 40, 120, 160))],
        )
        second = tracker.update(
            self.frame,
            [raw_track(999, BoundingBox(25, 42, 125, 162))],
        )

        self.assertEqual(first[0].track_id, 1)
        self.assertEqual(second[0].track_id, 1)

    def test_nearby_identities_are_matched_one_to_one(self) -> None:
        tracker = PersistentIdentityTracker(MeanColorEmbedding())
        first = tracker.update(
            self.frame,
            [
                raw_track(10, BoundingBox(20, 40, 120, 160)),
                raw_track(20, BoundingBox(260, 40, 360, 160)),
            ],
        )
        second = tracker.update(
            self.frame,
            [
                raw_track(30, BoundingBox(25, 40, 125, 160)),
                raw_track(40, BoundingBox(255, 40, 355, 160)),
            ],
        )

        self.assertEqual([track.track_id for track in first], [1, 2])
        self.assertEqual([track.track_id for track in second], [1, 2])

    def test_expired_identity_number_is_recycled(self) -> None:
        tracker = PersistentIdentityTracker(
            MeanColorEmbedding(),
            max_missing_frames=2,
        )
        tracker.update(self.frame, [raw_track(10, BoundingBox(20, 40, 120, 160))])
        tracker.update(self.frame, [])
        tracker.update(self.frame, [])
        tracker.update(self.frame, [])

        result = tracker.update(
            self.frame,
            [raw_track(20, BoundingBox(20, 40, 120, 160))],
        )

        self.assertEqual(result[0].track_id, 1)
