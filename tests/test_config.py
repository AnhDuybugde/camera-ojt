from pathlib import Path
from unittest import TestCase

from camera_tracking.config import load_config


class ConfigTest(TestCase):
    def test_load_config(self) -> None:
        config = load_config(Path("config/default.yaml"))

        self.assertEqual(config.camera.source, 0)
        self.assertEqual(config.detection.model_path, "yolo11n.pt")
        self.assertEqual(len(config.analytics.calibration.image_points), 4)
