from __future__ import annotations

import threading

import cv2
import numpy as np

from camera_tracking.streaming.mjpeg import MjpegStreamer


class LatestJpegRenderer:
    """Encode and publish only the newest frame per channel in a worker."""

    def __init__(self, streamer: MjpegStreamer, quality: int = 80) -> None:
        self.streamer = streamer
        self.quality = max(1, min(100, quality))
        self._frames: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="mjpeg-render-worker", daemon=True
        )
        self._thread.start()

    def submit(self, name: str, frame: np.ndarray) -> None:
        with self._lock:
            self._frames[name] = frame
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.2)
            self._wake.clear()
            with self._lock:
                frames = self._frames
                self._frames = {}
            for name, frame in frames.items():
                ok, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                if ok:
                    self.streamer.push(name, buffer.tobytes())

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


__all__ = ["LatestJpegRenderer"]
