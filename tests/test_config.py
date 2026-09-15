from pathlib import Path
from unittest import TestCase

from camera_tracking.config import load_config


class ConfigTest(TestCase):
    def test_load_config(self) -> None:
        config = load_config(Path("config/default.yaml"))

        self.assertEqual(config.camera.source, 0)
        self.assertEqual(config.detection.model_path, "yolo26s.engine")
        self.assertEqual(config.detection.confidence_threshold, 0.25)
        self.assertEqual(config.detection.image_size, 640)
        self.assertEqual(config.detection.nms_iou_threshold, 0.50)
        self.assertEqual(config.tracking.min_hits, 3)
        self.assertEqual(config.tracking.track_high_threshold, 0.50)
        self.assertEqual(config.tracking.track_low_threshold, 0.10)
        self.assertEqual(config.tracking.new_track_threshold, 0.60)
        self.assertEqual(config.tracking.byte_match_threshold, 0.80)
        self.assertEqual(config.tracking.track_buffer, 90)
        self.assertEqual(config.identity.match_threshold, 0.70)
        self.assertEqual(config.identity.min_appearance_similarity, 0.70)
        self.assertEqual(config.identity.named_appearance_floor, 0.70)
        self.assertEqual(config.identity.active_duplicate_similarity, 0.90)
        self.assertEqual(config.identity.gallery_refresh_steps, 10)
        self.assertEqual(config.identity.reid_model, "osnet_x0_25")
        self.assertEqual(config.identity.tentative_min_hits, 5)
        self.assertEqual(config.face.match_threshold, 0.60)
        self.assertEqual(config.face.min_margin, 0.05)
        self.assertEqual(config.face.consensus_hits, 2)
        self.assertEqual(config.face.det_size, 320)
        self.assertEqual(config.face.min_face_score, 0.5)
        self.assertEqual(config.face.min_person_area_px, 8000.0)
        self.assertEqual(list(config.face.channels), ["A", "B"])
        self.assertEqual(config.face.face_device, "cuda")
        self.assertEqual(config.attendance.debounce_hits, 2)
        self.assertEqual(config.room_fusion.inroom_min_interval_s, 600.0)
        self.assertEqual(config.room_fusion.absent_fallback_s, 900.0)
        self.assertEqual(config.workstations, [])
        self.assertEqual(config.workstate.grace_s, 1.5)
        self.assertEqual(config.workstate.move_ratio, 0.0)
        self.assertEqual(config.voice.backend, "imou_p2p")
        self.assertEqual(config.voice.command_ttl_s, 5.0)
        self.assertEqual(config.voice.bridge_port, 8767)
        self.assertEqual(config.voice.p2p_channel, 1)
        self.assertEqual(config.voice.p2p_sample_rate, 16000)
        self.assertEqual(len(config.analytics.calibration.image_points), 4)
