from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from camera_tracking.domain import Frame


class OpenCVFrameSource:
    """Read a webcam, video file, or RTSP stream as timestamped frames."""

    def __init__(
        self,
        source: int | str,
        width: int | None = None,
        height: int | None = None,
        requested_fps: float | None = None,
        process_every_n_frames: int = 1,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.requested_fps = requested_fps
        self.process_every_n_frames = max(1, process_every_n_frames)
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> OpenCVFrameSource:
        source: int | str = self.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        elif isinstance(source, str) and "://" not in source:
            source = str(Path(source).expanduser())

        capture = cv2.VideoCapture(source)
        if self.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.requested_fps:
            capture.set(cv2.CAP_PROP_FPS, self.requested_fps)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open camera/video source: {self.source}")
        self._capture = capture
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def fps(self) -> float:
        if self._capture is None:
            return self.requested_fps or 25.0
        value = self._capture.get(cv2.CAP_PROP_FPS)
        return value if value > 0 else (self.requested_fps or 25.0)

    def __iter__(self) -> Iterator[Frame]:
        if self._capture is None:
            raise RuntimeError("Frame source must be opened with a context manager")

        raw_index = 0
        output_index = 0
        while True:
            ok, image = self._capture.read()
            if not ok:
                break
            if raw_index % self.process_every_n_frames == 0:
                timestamp_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_s = timestamp_ms / 1000.0 if timestamp_ms > 0 else raw_index / self.fps
                yield Frame(index=output_index, timestamp_s=timestamp_s, image=image)
                output_index += 1
            raw_index += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

