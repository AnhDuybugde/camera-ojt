from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from camera_tracking.domain import AnalyticsSnapshot


class ResultWriter:
    def __init__(
        self,
        output_dir: Path,
        save_video: bool,
        save_report: bool,
        video_filename: str,
        report_filename: str,
        fps: float,
    ) -> None:
        self.output_dir = output_dir
        self.save_video = save_video
        self.save_report = save_report
        self.video_path = output_dir / video_filename
        self.report_path = output_dir / report_filename
        self.fps = fps
        self._video: cv2.VideoWriter | None = None

    def write_frame(self, image: np.ndarray) -> None:
        if not self.save_video:
            return
        if self._video is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            height, width = image.shape[:2]
            self._video = cv2.VideoWriter(
                str(self.video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
            if not self._video.isOpened():
                raise RuntimeError(f"Cannot create output video: {self.video_path}")
        self._video.write(image)

    def close(self, snapshot: AnalyticsSnapshot | None) -> None:
        if self._video is not None:
            self._video.release()
            self._video = None
        if self.save_report and snapshot is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "last_frame": snapshot.frame_index,
                "duration_s": snapshot.timestamp_s,
                "occupancy": snapshot.occupancy,
                "entries": snapshot.entries,
                "exits": snapshot.exits,
                "density_grid": snapshot.density_grid,
                "heatmap_grid": snapshot.heatmap_grid,
                "zone_counts": snapshot.zone_counts,
                "floor_positions_m": {
                    str(key): value for key, value in snapshot.floor_positions.items()
                },
                "trajectories_m": {
                    str(key): value for key, value in snapshot.trajectories.items()
                },
                "movement_vectors_m": {
                    str(key): value for key, value in snapshot.movement_vectors.items()
                },
            }
            self.report_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
            )
