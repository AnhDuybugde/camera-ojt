from __future__ import annotations

from collections.abc import Iterable

import cv2

from camera_tracking.analytics import CountingLine, FloorProjector, RoomAnalytics, Zone
from camera_tracking.config import AppConfig
from camera_tracking.detection import PersonDetector, YoloPersonDetector
from camera_tracking.domain import AnalyticsSnapshot, Frame, TrackEvent
from camera_tracking.output import ResultWriter
from camera_tracking.runtime import StageMetrics, TrackEventBus
from camera_tracking.tracking import IoUTracker
from camera_tracking.visualization import OverlayRenderer


class CameraTrackingPipeline:
    def __init__(
        self,
        detector: PersonDetector,
        tracker: IoUTracker,
        analytics: RoomAnalytics,
        renderer: OverlayRenderer,
        writer: ResultWriter,
        display: bool = False,
        event_bus: TrackEventBus | None = None,
        metrics: StageMetrics | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.analytics = analytics
        self.renderer = renderer
        self.writer = writer
        self.display = display
        self.event_bus = event_bus
        self.metrics = metrics or StageMetrics()

    def run(
        self, frames: Iterable[Frame], max_frames: int | None = None
    ) -> AnalyticsSnapshot | None:
        last_snapshot: AnalyticsSnapshot | None = None
        window_created = False
        if self.display and not opencv_has_gui_support():
            raise RuntimeError(
                "OpenCV was installed without GUI support. Remove opencv-python-headless "
                "and opencv-contrib-python, then reinstall opencv-python. Alternatively, "
                "run without --display and inspect output/annotated.mp4."
            )
        try:
            for processed_count, frame in enumerate(frames, start=1):
                with self.metrics.measure("detection"):
                    detections = self.detector.detect(frame.image)
                with self.metrics.measure("tracking"):
                    tracks = self.tracker.update(detections)
                if self.event_bus is not None:
                    self.event_bus.publish(
                        TrackEvent(
                            channel="default",
                            frame=frame,
                            tracks=tuple(tracks),
                        )
                    )
                with self.metrics.measure("analytics"):
                    last_snapshot = self.analytics.update(
                        frame.index, frame.timestamp_s, tracks
                    )
                with self.metrics.measure("rendering"):
                    annotated = self.renderer.render(frame.image, tracks, last_snapshot)
                with self.metrics.measure("storage"):
                    self.writer.write_frame(annotated)

                if self.display:
                    try:
                        cv2.imshow("Camera tracking", annotated)
                        window_created = True
                    except cv2.error as error:
                        raise RuntimeError(
                            "OpenCV cannot create a display window. Run without --display "
                            "or install an OpenCV build with GUI support."
                        ) from error
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if max_frames is not None and processed_count >= max_frames:
                    break
        finally:
            self.writer.close(last_snapshot)
            if window_created:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
        return last_snapshot


def opencv_has_gui_support() -> bool:
    for line in cv2.getBuildInformation().splitlines():
        normalized = line.strip().upper()
        if normalized.startswith("GUI:"):
            return not normalized.endswith("NONE")
    return False


def build_pipeline(config: AppConfig) -> CameraTrackingPipeline:
    projector = FloorProjector(
        config.analytics.calibration.image_points,
        config.analytics.calibration.floor_points,
    )
    line_config = config.analytics.counting_line
    counting_line = (
        CountingLine(
            start=line_config.start,
            end=line_config.end,
            entry_direction=line_config.entry_direction,
        )
        if line_config
        else None
    )
    zones = [Zone(name=zone.name, points=zone.points) for zone in config.analytics.zones]
    analytics = RoomAnalytics(
        projector=projector,
        floor_width_m=config.analytics.floor_width_m,
        floor_height_m=config.analytics.floor_height_m,
        grid_rows=config.analytics.density_grid.rows,
        grid_cols=config.analytics.density_grid.cols,
        zones=zones,
        counting_line=counting_line,
        trajectory_length=config.analytics.trajectory_length,
    )
    output_fps = config.camera.fps / config.camera.process_every_n_frames
    return CameraTrackingPipeline(
        detector=YoloPersonDetector(
            model_path=config.detection.model_path,
            confidence=config.detection.confidence_threshold,
            person_class_id=config.detection.person_class_id,
            image_size=config.detection.image_size,
            device=config.detection.device,
            nms_iou_threshold=config.detection.nms_iou_threshold,
            nested_box_containment_threshold=(
                config.detection.nested_box_containment_threshold
            ),
        ),
        tracker=IoUTracker(
            iou_threshold=config.tracking.iou_threshold,
            max_lost_frames=config.tracking.max_lost_frames,
            min_hits=config.tracking.min_hits,
        ),
        analytics=analytics,
        renderer=OverlayRenderer(
            projector=projector,
            floor_width_m=config.analytics.floor_width_m,
            floor_height_m=config.analytics.floor_height_m,
            zones=zones,
            counting_line=counting_line,
        ),
        writer=ResultWriter(
            output_dir=config.output.output_dir,
            save_video=config.output.save_video,
            save_report=config.output.save_report,
            video_filename=config.output.video_filename,
            report_filename=config.output.report_filename,
            fps=output_fps,
        ),
        display=config.output.display,
    )
