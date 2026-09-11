from __future__ import annotations

from unittest import TestCase

import numpy as np

from camera_tracking.domain import BoundingBox, Track
from camera_tracking.visualization import draw_person_tracks


class PersonOverlayTest(TestCase):
    def test_draws_box_label_and_count_without_resizing_frame(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        track = Track(
            track_id=7,
            bbox=BoundingBox(80, 70, 180, 210),
            confidence=0.91,
            age=2,
            hits=2,
            confirmed=True,
        )

        rendered = draw_person_tracks(frame, [track])

        self.assertIs(rendered, frame)
        self.assertEqual(rendered.shape, (240, 320, 3))
        self.assertGreater(np.count_nonzero(rendered), 0)
        self.assertTrue(np.any(rendered[70, 80] != 0))
