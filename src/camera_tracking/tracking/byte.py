from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from camera_tracking.domain import BoundingBox, Detection, Track


class ByteTrackTracker:
    """Adapter around Ultralytics ByteTrack with the project's domain objects."""

    def __init__(
        self,
        *,
        frame_rate: int = 12,
        track_high_threshold: float = 0.25,
        track_low_threshold: float = 0.10,
        new_track_threshold: float = 0.25,
        track_buffer: int = 30,
        match_threshold: float = 0.8,
        min_hits: int = 2,
    ) -> None:
        try:
            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as error:
            raise RuntimeError(
                "ByteTrack dependencies are missing. Run: pip install -r requirements.txt"
            ) from error

        args = SimpleNamespace(
            track_high_thresh=track_high_threshold,
            track_low_thresh=track_low_threshold,
            new_track_thresh=new_track_threshold,
            track_buffer=track_buffer,
            match_thresh=match_threshold,
            fuse_score=True,
        )
        self._tracker = BYTETracker(args, frame_rate=max(1, frame_rate))
        self._boxes_type = Boxes
        self.min_hits = min_hits
        self._ages: dict[int, int] = {}
        self._hits: dict[int, int] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        values = np.asarray(
            [
                [
                    detection.bbox.x1,
                    detection.bbox.y1,
                    detection.bbox.x2,
                    detection.bbox.y2,
                    detection.confidence,
                    detection.class_id,
                ]
                for detection in detections
            ],
            dtype=np.float32,
        ).reshape(-1, 6)
        boxes = self._boxes_type(values, orig_shape=(1, 1))
        tracked = self._tracker.update(boxes)

        for track_id in self._ages:
            self._ages[track_id] += 1

        output: list[Track] = []
        for row in tracked:
            track_id = int(row[4])
            self._ages.setdefault(track_id, 1)
            self._hits[track_id] = self._hits.get(track_id, 0) + 1
            output.append(
                Track(
                    track_id=track_id,
                    bbox=BoundingBox(*(float(value) for value in row[:4])),
                    confidence=float(row[5]),
                    age=self._ages[track_id],
                    hits=self._hits[track_id],
                    confirmed=self._hits[track_id] >= self.min_hits,
                )
            )
        return output
