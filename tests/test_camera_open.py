from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from scripts.run_workstate import _substream_fallback, normalize_frame_size, open_capture


class OpenCaptureTest(TestCase):
    def test_main_stream_has_substream_fallback(self) -> None:
        source = "rtsp://user:secret@camera/live?channel=1&subtype=0"

        fallback = _substream_fallback(source)

        self.assertEqual(
            fallback,
            "rtsp://user:secret@camera/live?channel=1&subtype=1",
        )

    def test_frame_is_resized_to_roi_coordinate_system(self) -> None:
        frame = np.zeros((1620, 2880, 3), dtype=np.uint8)

        resized = normalize_frame_size(frame, 1280, 720)

        self.assertEqual(resized.shape, (720, 1280, 3))

    @patch("scripts.run_workstate.time.sleep")
    @patch("scripts.run_workstate.cv2.VideoCapture")
    def test_rtsp_source_retries_until_opened(
        self, video_capture: MagicMock, sleep: MagicMock
    ) -> None:
        capture = MagicMock()
        capture.isOpened.side_effect = [False, False, True]
        video_capture.return_value = capture

        result = open_capture("rtsp://camera/stream", attempts=3, retry_delay_s=0.25)

        self.assertIs(result, capture)
        self.assertEqual(capture.open.call_count, 3)
        self.assertEqual(capture.release.call_count, 2)
        self.assertEqual(sleep.call_count, 2)

    @patch("scripts.run_workstate.cv2.VideoCapture")
    def test_video_file_is_not_retried(self, video_capture: MagicMock) -> None:
        capture = MagicMock()
        capture.isOpened.return_value = False
        video_capture.return_value = capture

        open_capture("data/sample.mp4", attempts=5)

        capture.open.assert_called_once()

    @patch("scripts.run_workstate.time.sleep")
    @patch("scripts.run_workstate.cv2.VideoCapture")
    def test_main_stream_falls_back_to_substream(
        self, video_capture: MagicMock, sleep: MagicMock
    ) -> None:
        capture = MagicMock()
        capture.isOpened.side_effect = [False, True]
        video_capture.return_value = capture

        result = open_capture(
            "rtsp://camera/live?channel=1&subtype=0",
            attempts=1,
            retry_delay_s=0,
        )

        self.assertIs(result, capture)
        self.assertEqual(capture.open.call_count, 2)
        fallback_source = capture.open.call_args.args[0]
        self.assertIn("subtype=1", fallback_source)
        sleep.assert_not_called()
