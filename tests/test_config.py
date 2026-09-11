from pathlib import Path
from unittest import TestCase

from camera_tracking.config import load_config


class ConfigTest(TestCase):
    def test_load_config(self) -> None:
        config = load_config(Path("config/default.yaml"))

        self.assertEqual(config.camera.source, 0)
        self.assertEqual(config.detection.model_path, "yolo26s.pt")
        self.assertEqual(config.detection.confidence_threshold, 0.25)
        self.assertEqual(config.detection.image_size, 800)
        self.assertEqual(len(config.analytics.calibration.image_points), 4)
