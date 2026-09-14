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
    confidence_threshold: float = Field(default=0.4, ge=0, le=1)
    nms_iou_threshold: float = Field(default=0.50, ge=0, le=1)
    nested_box_containment_threshold: float = Field(default=0.85, ge=0, le=1)
    person_class_id: int = 0
    image_size: int = 800
    device: str = "auto"


class TrackingConfig(StrictModel):
    max_lost_frames: int = Field(default=15, ge=0)
    min_hits: int = Field(default=3, ge=1)
    iou_threshold: float = Field(default=0.3, ge=0, le=1)
    byte_match_threshold: float = Field(default=0.50, ge=0, le=1)
    track_buffer: int = Field(default=30, ge=1)


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
    # Gate rieng cho ID da co ten (employee_id): muon tai su dung GID nay,
    # appearance phai >= nguong nay ke ca khi spatial/time cao. Chong vu
    # G1-LeHoAnhDuy an di roi gán sang nguoi khac dung gan.
    named_appearance_floor: float = Field(default=0.45, ge=0, le=1)
    active_duplicate_similarity: float = Field(default=0.60, ge=0, le=1)
    temp_lost_s: float = Field(default=5.0, ge=0)
    long_lost_s: float = Field(default=60.0, ge=0)
    unresolved_keep_s: float = Field(default=300.0, ge=0)
    min_gallery_confidence: float = Field(default=0.25, ge=0, le=1)
    gallery_refresh_steps: int = Field(default=1, ge=1)
    # Backend appearance cho Global ID: "face" (khuyen nghi: nhan dang
    # bang mat, quay lung thi khong ep match), "osnet" (body), "histogram".
    reid_backend: Literal["face", "osnet", "histogram"] = "face"
    reid_model: str = "osnet_x1_0"
    reid_device: Literal["auto", "cpu", "cuda"] = "auto"
    # Mat nho hon nguong nay thi ReID coi nhu "khong thay mat".
    face_min_px: int = Field(default=40, ge=8)


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


class FaceConfig(StrictModel):
    """Nhan dien khuon mat (diem danh). Chay tren cac channel chi dinh."""

    enabled: bool = True
    # Channel chay face recognition/diem danh, vi du ["B"] (cua) hoac
    # ["A", "B"] (ca 2 camera nhu nhau).
    channels: list[str] = Field(default_factory=lambda: ["A", "B"])
    # Thiet bi InsightFace: auto = cuda neu co CUDAExecutionProvider (can
    # onnxruntime-gpu + CUDA Toolkit), khong thi cpu.
    face_device: Literal["auto", "cpu", "cuda"] = "auto"
    gallery_dir: Path = Path("data/images")
    model_pack: str = "buffalo_s"
    match_threshold: float = Field(default=0.3, ge=0, le=1)
    min_face_px: int = Field(default=32, ge=8)
    min_blur_variance: float = Field(default=20.0, ge=0)
    min_person_area_px: float = Field(default=2000.0, ge=0)
    process_every_k: int = Field(default=3, ge=1)
    det_size: int = Field(default=640, ge=160)
    # Ten hien thi: {"LeHoAnhDuy": "Le Ho Anh Duy"}; mac dinh dung stem file.
    name_map: dict[str, str] = Field(default_factory=dict)
    # Employee ID trong database: {"LeHoAnhDuy": "1"}.
    employee_map: dict[str, str] = Field(default_factory=dict)


class AttendanceConfig(StrictModel):
    """Debounce tick diem danh: k hit cung person trong window moi tick."""

    debounce_hits: int = Field(default=1, ge=1)
    window_s: float = Field(default=8.0, ge=0)
    # Chi tick trong khung gio nay (gio dia phuong, 24h). None = ca ngay.
    # Vi du lam viec: start 6, end 22.
    active_hour_start: int | None = Field(default=None, ge=0, le=23)
    active_hour_end: int | None = Field(default=None, ge=0, le=23)


class RoomFusionConfig(StrictModel):
    """Fuse trang thai phong tu channel A (room) + B (door)."""

    leave_confirm_window_s: float = Field(default=300.0, ge=0)
    inroom_min_interval_s: float = Field(default=3600.0, ge=0)
    flush_s: float = Field(default=15.0, ge=1)


class WorkstationConfig(StrictModel):
    """One desk/seat: core + extended polygons in FLOOR METERS.

    Same coordinate frame as analytics.calibration floor_points
    (homography maps bbox foot points into it).
    """

    name: str
    core: list[Point]
    extended: list[Point]


class WorkstateConfig(StrictModel):
    """Position-based work state (replaces motion-based seat logic)."""

    grace_s: float = Field(default=3.0, ge=0)
    dwell_s: float = Field(default=2.0, ge=0)
    assign_dwell_s: float = Field(default=5.0, ge=0)
    hysteresis_m: float = Field(default=0.3, ge=0)
    motion_influence: float = Field(default=0.0, ge=0, le=1)
    prune_after_s: float = Field(default=300.0, ge=0)
    tentative_min_hits: int = Field(default=5, ge=1)
    person_map: dict[str, str] = Field(default_factory=dict)
    # INTERIM (no fixed camera / no ROI yet): normalized 0..1 bbox-center
    # displacement from the first-seen anchor counting as leave-seat.
    # 0 disables -> pure presence mode.
    move_ratio: float = Field(default=0.15, ge=0, le=1)
    settle_ratio: float = Field(default=0.02, ge=0, le=1)


class StoreConfig(StrictModel):
    """Luu tru daily: local faces + Supabase (optional, bat khi co .env)."""

    local_faces_dir: Path = Path("data/faces")
    queue_db: Path = Path("output/queue.db")
    supabase_enabled: bool = False
    max_crops_per_owner_day: int = Field(default=5, ge=1)
    face_jpeg_quality: int = Field(default=80, ge=10, le=100)
    face_max_side_px: int = Field(default=512, ge=64)


class AppConfig(StrictModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    identity: GlobalIdentityConfig = Field(default_factory=GlobalIdentityConfig)
    analytics: AnalyticsConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    face: FaceConfig = Field(default_factory=FaceConfig)
    attendance: AttendanceConfig = Field(default_factory=AttendanceConfig)
    room_fusion: RoomFusionConfig = Field(default_factory=RoomFusionConfig)
    workstations: list[WorkstationConfig] = Field(default_factory=list)
    workstate: WorkstateConfig = Field(default_factory=WorkstateConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {config_path}")
    return AppConfig.model_validate(data)
