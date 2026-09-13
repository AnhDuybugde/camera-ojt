from pathlib import Path
from unittest import TestCase

from camera_tracking.config import load_config


class ConfigTest(TestCase):
    def test_load_config(self) -> None:
        config = load_config(Path("config/default.yaml"))

        self.assertEqual(config.camera.source, 0)
        self.assertEqual(config.detection.model_path, "yolo26s.pt")
        self.assertEqual(config.detection.confidence_threshold, 0.4)
        self.assertEqual(config.detection.image_size, 800)
        self.assertEqual(config.detection.nms_iou_threshold, 0.50)
        self.assertEqual(config.tracking.min_hits, 3)
        self.assertEqual(config.workstations, [])
        self.assertEqual(config.workstate.grace_s, 1.5)
        self.assertEqual(config.workstate.tentative_min_hits, 5)
        self.assertEqual(config.workstate.move_ratio, 0.15)
        self.assertEqual(len(config.analytics.calibration.image_points), 4)
