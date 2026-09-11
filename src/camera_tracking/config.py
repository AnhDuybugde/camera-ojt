from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from camera_tracking.domain import Point


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CameraConfig(StrictModel):
    source: int | str = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: float = 25
    process_every_n_frames: int = Field(default=2, ge=1)


class DetectionConfig(StrictModel):
    model_path: str = "yolo26s.pt"
    confidence_threshold: float = Field(default=0.35, ge=0, le=1)
    person_class_id: int = 0
    image_size: int = 800
    device: str = "auto"


class TrackingConfig(StrictModel):
    max_lost_frames: int = Field(default=15, ge=0)
    min_hits: int = Field(default=2, ge=1)
    iou_threshold: float = Field(default=0.3, ge=0, le=1)


class GlobalIdentityConfig(StrictModel):
    """Cau hinh Global Identity Manager (xem tracking/global_identity.py)."""

    gallery_size: int = Field(default=8, ge=1)
    appearance_weight: float = Field(default=0.55, ge=0)
    spatial_weight: float = Field(default=0.20, ge=0)
    time_weight: float = Field(default=0.15, ge=0)
    channel_weight: float = Field(default=0.10, ge=0)
    match_threshold: float = Field(default=0.40, ge=0, le=1)
    max_center_distance_ratio: float = Field(default=0.35, gt=0)
    min_appearance_similarity: float = Field(default=0.30, ge=0, le=1)
    temp_lost_s: float = Field(default=5.0, ge=0)
    long_lost_s: float = Field(default=60.0, ge=0)
    unresolved_keep_s: float = Field(default=300.0, ge=0)
    min_gallery_confidence: float = Field(default=0.25, ge=0, le=1)


class CalibrationConfig(StrictModel):
    image_points: list[Point]
    floor_points: list[Point]

    @model_validator(mode="after")
    def validate_point_pairs(self) -> CalibrationConfig:
        if len(self.image_points) != len(self.floor_points) or len(self.image_points) < 4:
            raise ValueError("calibration needs at least four matching point pairs")
        return self


class DensityGridConfig(StrictModel):
    rows: int = Field(default=4, ge=1)
    cols: int = Field(default=6, ge=1)


class ZoneConfig(StrictModel):
    name: str
    points: list[Point]


class CountingLineConfig(StrictModel):
    start: Point
    end: Point
    entry_direction: Literal["negative_to_positive", "positive_to_negative"] = (
        "negative_to_positive"
    )


class AnalyticsConfig(StrictModel):
    floor_width_m: float = Field(gt=0)
    floor_height_m: float = Field(gt=0)
    trajectory_length: int = Field(default=120, ge=2)
    calibration: CalibrationConfig
    density_grid: DensityGridConfig = Field(default_factory=DensityGridConfig)
    zones: list[ZoneConfig] = Field(default_factory=list)
    counting_line: CountingLineConfig | None = None


class OutputConfig(StrictModel):
    output_dir: Path = Path("output")
    save_video: bool = True
    save_report: bool = True
    display: bool = False
    video_filename: str = "annotated.mp4"
    report_filename: str = "report.json"


class AppConfig(StrictModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    identity: GlobalIdentityConfig = Field(default_factory=GlobalIdentityConfig)
    analytics: AnalyticsConfig
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {config_path}")
    return AppConfig.model_validate(data)
