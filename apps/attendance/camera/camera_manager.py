"""Non-blocking webcam/RTSP capture with reconnect support."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import cv2
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class CameraManager:
    def __init__(self, source: int | str | None = None) -> None:
        self.source = settings.camera_source if source is None else source
        self._capture: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False
        self.error = ""

    def _open(self) -> bool:
        if isinstance(self.source, str):
            # TCP is more reliable for office LANs; timeouts prevent a dead RTSP endpoint
            # from blocking the reconnect worker indefinitely.
            transport = settings.rtsp_transport if settings.rtsp_transport in {"tcp", "udp"} else "tcp"
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
            capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            capture = cv2.VideoCapture(self.source)
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5_000)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(self.source, int):
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
        if not capture.isOpened():
            capture.release()
            self.error = f"Cannot open camera: {self.safe_source}"
            return False
        self._capture = capture
        self.connected = True
        self.error = ""
        logger.info("Camera connected: %s", self.safe_source)
        return True

    @property
    def safe_source(self) -> str:
        if not isinstance(self.source, str) or "://" not in self.source:
            return str(self.source)
        scheme, remainder = self.source.split("://", 1)
        return f"{scheme}://***@{remainder.split('@', 1)[-1]}"

    def start(self) -> "CameraManager":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="camera-capture")
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            if self._capture is None or not self._capture.isOpened():
                if not self._open():
                    self._stop.wait(settings.camera_reconnect_delay)
                    continue
            ok, frame = self._capture.read()
            if ok and frame is not None:
                failures = 0
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.monotonic()
                continue
            failures += 1
            if failures >= 10:
                logger.warning("Camera disconnected: %s", self.safe_source)
                self.error = "Camera connection lost; reconnecting..."
                self.connected = False
                self._release_capture()
                failures = 0
                self._stop.wait(settings.camera_reconnect_delay)
            else:
                time.sleep(0.03)
        self._release_capture()

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            return (self._frame is not None, None if self._frame is None else self._frame.copy())

    @property
    def frame_age_ms(self) -> int | None:
        with self._lock:
            if not self._frame_time:
                return None
            return max(0, int((time.monotonic() - self._frame_time) * 1000))

    def wait_for_frame(self, timeout: float = 8.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._stop.is_set():
            if self.read()[0]:
                return True
            time.sleep(0.05)
        return False

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.connected = False

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._release_capture()

    def release(self) -> None:
        self.stop()

    def isOpened(self) -> bool:  # OpenCV-compatible registration interface
        return self.connected

    def __enter__(self) -> "CameraManager":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()
