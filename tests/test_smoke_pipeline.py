from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from camera_tracking.analytics import CountingLine, FloorProjector, RoomAnalytics, Zone
from camera_tracking.domain import BoundingBox, Detection, Frame
from camera_tracking.output import ResultWriter
from camera_tracking.pipeline import CameraTrackingPipeline, opencv_has_gui_support
from camera_tracking.tracking import IoUTracker
from camera_tracking.visualization import OverlayRenderer


class MovingPersonDetector:
    def __init__(self) -> None:
        self.frame_index = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        foot_y = 70 + self.frame_index * 20
        self.frame_index += 1
        return [
            Detection(
                bbox=BoundingBox(120, foot_y - 40, 160, foot_y),
                confidence=0.9,
            )
        ]


class PipelineSmokeTest(TestCase):
    def test_headless_opencv_is_detected(self) -> None:
        with patch("cv2.getBuildInformation", return_value="  GUI: NONE\n"):
            self.assertFalse(opencv_has_gui_support())

        with patch("cv2.getBuildInformation", return_value="  GUI: WIN32UI\n"):
            self.assertTrue(opencv_has_gui_support())

    def test_all_stages_and_floor_trajectory(self) -> None:
        projector = FloorProjector(
            image_points=[(0, 0), (320, 0), (320, 240), (0, 240)],
            floor_points=[(0, 0), (320, 0), (320, 240), (0, 240)],
        )
        analytics = RoomAnalytics(
            projector=projector,
            floor_width_m=320,
            floor_height_m=240,
            grid_rows=2,
            grid_cols=2,
            zones=[Zone("room", [(0, 0), (320, 0), (320, 240), (0, 240)])],
            counting_line=CountingLine((0, 110), (320, 110)),
            trajectory_length=10,
        )
        frames = [
            Frame(index=index, timestamp_s=index / 5, image=np.zeros((240, 320, 3), np.uint8))
            for index in range(4)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ResultWriter(
                output_dir=Path(temp_dir),
                save_video=False,
                save_report=True,
                video_filename="smoke.mp4",
                report_filename="smoke.json",
                fps=5,
            )
            pipeline = CameraTrackingPipeline(
                detector=MovingPersonDetector(),
                tracker=IoUTracker(iou_threshold=0.3, max_lost_frames=2, min_hits=1),
                analytics=analytics,
                renderer=OverlayRenderer(
                    projector,
                    floor_width_m=320,
                    floor_height_m=240,
                    zones=analytics.zones,
                    counting_line=analytics.counting_line,
                ),
                writer=writer,
            )
            snapshot = pipeline.run(frames)

            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.occupancy, 1)
            self.assertEqual(snapshot.entries, 1)
            self.assertEqual(snapshot.exits, 0)
            self.assertEqual(snapshot.zone_counts["room"], 1)
            self.assertEqual(len(snapshot.trajectories[1]), 4)
            self.assertEqual(snapshot.floor_positions[1], (140.0, 130.0))
            self.assertEqual(sum(map(sum, snapshot.heatmap_grid)), 4)
            self.assertEqual(snapshot.movement_vectors[1], (0.0, 20.0))

            report = json.loads((Path(temp_dir) / "smoke.json").read_text("utf-8"))
            self.assertEqual(report["entries"], 1)
            self.assertEqual(len(report["trajectories_m"]["1"]), 4)
