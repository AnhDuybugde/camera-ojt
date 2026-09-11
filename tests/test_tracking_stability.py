from unittest import TestCase

from camera_tracking.tracking import StablePersonCount


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
