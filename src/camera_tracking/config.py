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
    confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    nms_iou_threshold: float = Field(default=0.85, ge=0, le=1)
    nested_box_containment_threshold: float = Field(default=0.85, ge=0, le=1)
    person_class_id: int = 0
    image_size: int = 640
    device: str = "auto"


class TrackingConfig(StrictModel):
    max_lost_frames: int = Field(default=15, ge=0)
    min_hits: int = Field(default=3, ge=1)
    iou_threshold: float = Field(default=0.3, ge=0, le=1)
    track_high_threshold: float = Field(default=0.50, ge=0, le=1)
    track_low_threshold: float = Field(default=0.10, ge=0, le=1)
    new_track_threshold: float = Field(default=0.60, ge=0, le=1)
    byte_match_threshold: float = Field(default=0.80, ge=0, le=1)
    track_buffer: int = Field(default=90, ge=1)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> TrackingConfig:
        if self.track_low_threshold > self.track_high_threshold:
            raise ValueError("track_low_threshold must not exceed track_high_threshold")
        if self.new_track_threshold < self.track_high_threshold:
            raise ValueError("new_track_threshold must be at least track_high_threshold")
        return self


class GlobalIdentityConfig(StrictModel):
    """Cau hinh Global Identity Manager (xem tracking/global_identity.py)."""

    gallery_size: int = Field(default=8, ge=1)
    appearance_weight: float = Field(default=0.60, ge=0)
    spatial_weight: float = Field(default=0.15, ge=0)
    time_weight: float = Field(default=0.15, ge=0)
    channel_weight: float = Field(default=0.10, ge=0)
    match_threshold: float = Field(default=0.70, ge=0, le=1)
    max_center_distance_ratio: float = Field(default=0.35, gt=0)
    same_camera_reconnect_distance_ratio: float = Field(default=0.08, gt=0)
    same_camera_reconnect_s: float = Field(default=2.0, ge=0)
    min_appearance_similarity: float = Field(default=0.70, ge=0, le=1)
    # Gate rieng cho ID da co ten (employee_id): muon tai su dung GID nay,
    # appearance phai >= nguong nay ke ca khi spatial/time cao. Chong vu
    # G1-LeHoAnhDuy an di roi gán sang nguoi khac dung gan.
    named_appearance_floor: float = Field(default=0.70, ge=0, le=1)
    active_duplicate_similarity: float = Field(default=0.90, ge=0, le=1)
    temp_lost_s: float = Field(default=10.0, ge=0)
    long_lost_s: float = Field(default=120.0, ge=0)
    unresolved_keep_s: float = Field(default=86400.0, ge=0)
    min_gallery_confidence: float = Field(default=0.25, ge=0, le=1)
    gallery_refresh_steps: int = Field(default=5, ge=1)
    tentative_min_hits: int = Field(default=5, ge=1)
    state_db: Path = Path("output/identity_state.db")
    reid_model: str = "osnet_x0_25"
    reid_device: Literal["auto", "cpu", "cuda"] = "auto"


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
    match_threshold: float = Field(default=0.60, ge=0, le=1)
    min_margin: float = Field(default=0.05, ge=0, le=1)
    consensus_hits: int = Field(default=2, ge=1)
    consensus_window_s: float = Field(default=3.0, gt=0)
    gallery_accept_threshold: float = Field(default=0.70, ge=0, le=1)
    gallery_max_prototypes: int = Field(default=10, ge=1)
    min_face_px: int = Field(default=40, ge=8)
    min_face_score: float = Field(default=0.5, ge=0, le=1)
    min_blur_variance: float = Field(default=40.0, ge=0)
    min_person_area_px: float = Field(default=8000.0, ge=0)
    process_every_k: int = Field(default=3, ge=1)
    det_size: int = Field(default=320, ge=160)
    # Retry cadence cho Face async worker (event-driven once-per-track).
    # unknown: GID chua biet ten -> thu lai sau X giay de bat goc mat dep.
    # 0.6s (~7 processed frame @12fps) nhanh hon default cu 1.0s mot chut
    # nhung van nhe GPU; may yeu se tu tang len nho backpressure theo
    # queue occupancy trong run_workstate._enqueue_face_async.
    unknown_cooldown_s: float = Field(default=0.6, ge=0)
    # known: GID da biet ten -> recheck sau X giay. Voice/greet se lay
    # min(known, 2.0s) de co observation moi cho trigger tay+mat.
    known_cooldown_s: float = Field(default=30.0, ge=0)
    # Ten hien thi: {"LeHoAnhDuy": "Le Ho Anh Duy"}; mac dinh dung stem file.
    name_map: dict[str, str] = Field(default_factory=dict)
    # Employee ID trong database: {"LeHoAnhDuy": "1"}.
    employee_map: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_gallery_threshold(self) -> FaceConfig:
        if self.gallery_accept_threshold < self.match_threshold:
            raise ValueError("gallery_accept_threshold must be at least match_threshold")
        return self


class AttendanceConfig(StrictModel):
    """Debounce tick diem danh: k hit cung person trong window moi tick."""

    debounce_hits: int = Field(default=2, ge=1)
    window_s: float = Field(default=8.0, ge=0)
    # Chi tick trong khung gio nay (gio dia phuong, 24h). None = ca ngay.
    # Vi du lam viec: start 6, end 22.
    active_hour_start: int | None = Field(default=None, ge=0, le=23)
    active_hour_end: int | None = Field(default=None, ge=0, le=23)


class RoomFusionConfig(StrictModel):
    """Fuse trang thai phong tu channel A (room) + B (door)."""

    leave_confirm_window_s: float = Field(default=300.0, ge=0)
    inroom_min_interval_s: float = Field(default=600.0, ge=0)
    absent_fallback_s: float = Field(default=900.0, ge=0)
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
    person_map: dict[str, str] = Field(default_factory=dict)
    # INTERIM (no fixed camera / no ROI yet): normalized 0..1 bbox-center
    # displacement from the first-seen anchor counting as leave-seat.
    # 0 disables -> pure presence mode (khuyến nghị mặc định cho tới
    # khi ROI được đo thật).
    move_ratio: float = Field(default=0.0, ge=0, le=1)
    settle_ratio: float = Field(default=0.02, ge=0, le=1)


class VoiceConfig(StrictModel):
    """Chao bang giong noi khi co nguoi vay tay (MediaPipe + edge-tts).

    backend ``imou_p2p`` (mac dinh moi): TTS -> ffmpeg AAC -> P2P VisualTalk
    thang ra loa camera, khong can browser/WebSDK/APP_ID. ``imou_web`` giu
    lai de fallback legacy.
    """

    enabled: bool = False
    backend: Literal["local", "imou_web", "imou_p2p"] = "imou_p2p"
    voice: str = "vi-VN-HoaiMyNeural"
    cache_dir: Path = Path("output/voice_cache")
    # Thong nhat 10s cho ca quen lan la; chi kich hoat khi gio tay 5 ngon.
    cooldown_s: float = Field(default=10.0, ge=0)
    # Cooldown rieng cho nguoi la (giu de tuy chinh, mac dinh cung 10s).
    unknown_cooldown_s: float = Field(default=10.0, ge=0)
    # Palm cua nguoi chua co ket qua face: True = chao ngay "Xin chào quý khách",
    # khong can doi FaceWorker. False = hanh vi cu (bo qua + log hint).
    palm_unknown_immediate: bool = True
    unknown_phrase: str = "Xin chào quý khách"
    # TTL hang doi loi chao: luot cho qua han thi bo. Phat 1 cau mat vai
    # giay nen de 10s de nguoi den sau choi 1-2 luot van duoc chao.
    command_ttl_s: float = Field(default=10.0, gt=0)
    # Khoang lang chong spam sau moi lan phat cho 1 nguoi: trong khoang
    # nay yeu cau chu dong (vay tay / voice) cua chinh nguoi do bi bo qua.
    proactive_quiet_s: float = Field(default=5.0, ge=0)
    bridge_host: str = "127.0.0.1"
    bridge_port: int = Field(default=8767, ge=0, le=65535)
    talk_tail_s: float = Field(default=0.3, ge=0)
    launch_browser: bool = True
    # P2P VisualTalk (vendor tu test-sound-camera-imou).
    # p2p_channel = loa kenh A (phong), p2p_channel_b = loa kenh B (cua).
    # Pipeline route loi chao theo camera phat hien (A->1, B->2).
    p2p_channel: int = Field(default=1, ge=1)
    p2p_channel_b: int = Field(default=2, ge=1)
    p2p_timeout_s: float = Field(default=20.0, gt=0)
    p2p_attempts: int = Field(default=2, ge=1)
    p2p_retry_delay_s: float = Field(default=5.0, ge=0)
    p2p_sample_rate: int = Field(default=16000, ge=8000)
    p2p_volume: float = Field(default=0.1, ge=0.0, le=1.0)
    # WAV chao tao san bang ZeroTTS (scripts/build_greeting_wavs.py):
    # manifest.json anh xa nguyen van cau chao -> file wav.
    greeting_dir: Path = Path("output/voice_greetings")
    zerotts_voice: str = "maichi"
    zerotts_model: str = "zeroweight-ai/ZeroTTS"
    # Chào mặt (face-triggered): mac dinh TAT, chi chao khi gio tay 5 ngon
    # (palm-only, ap dung chung cho ca quen lan la). Muon dung truoc cam
    # la chao ngay thi bat greet_on_face=true.
    # Dùng chung VoiceGreeter queue/cooldown với wave để không spam.
    greet_on_face: bool = False
    greet_unknown_on_face: bool = False
    # Override theo kenh (A=phong, B=cua). None = fallback ve global o tren.
    # Mac dinh mong muon: A wave-only (khong chao mat), B face-trigger
    # (thay mat la chao ngay, khong can tay).
    greet_on_face_a: bool | None = None
    greet_unknown_on_face_a: bool | None = None
    greet_on_face_b: bool | None = None
    greet_unknown_on_face_b: bool | None = None
    # Gesture theo kenh: "wave" (vay tay, khuyen nghi cho A),
    # "palm" (gio tay, legacy), "off" (tat, khuyen nghi cho B).
    gesture_a: Literal["wave", "palm", "off"] = "wave"
    gesture_b: Literal["wave", "palm", "off"] = "off"
    # Wave detector cadence + gioi han tai (MediaPipe chay CPU).
    # Toan cadence: process ~12.5 frame/s (fps/2) / every_k = tan so lay mau
    # co tay; can >= min_reversals+2 diem trong window moi fire duoc.
    # every_k=3, window=2.5s -> ~10 diem/window (du cho min_reversals=2).
    # Legacy wave (giữ để tương thích test cũ).
    wave_every_k: int = Field(default=3, ge=1)
    wave_max_people: int = Field(default=2, ge=1)
    wave_window_s: float = Field(default=2.5, gt=0)
    wave_min_reversals: int = Field(default=2, ge=2)
    wave_min_amplitude: float = Field(default=0.06, gt=0)
    wave_min_gap_s: float = Field(default=0.08, gt=0)
    wave_cooldown_s: float = Field(default=30.0, ge=0)
    # Stable foreground: gesture + primary overlay only follow this subject.
    focus_switch_area_ratio: float = Field(default=1.20, ge=1.0)
    focus_switch_hold_s: float = Field(default=0.50, ge=0.0)
    focus_lost_grace_s: float = Field(default=1.00, ge=0.0)
    # A wave is intentional only after a full open-palm activation.
    wave_required_fingers: int = Field(default=5, ge=5, le=5)
    wave_palm_confirm_frames: int = Field(default=2, ge=1)
    wave_palm_release_frames: int = Field(default=5, ge=1)
    wave_face_ttl_s: float = Field(default=8.0, gt=0.0)
    wave_face_max_distance: float = Field(default=3.5, gt=0.0)
    face_background_aging_s: float = Field(default=5.0, gt=0.0)
    # Trigger hinh anh: mat phai nhin thang vao camera TRUOC roi moi toi
    # tay. Danh gia doi xung 5 diem detector (mui giua 2 mat = yaw,
    # 2 mat ngang nhau = roll), ti le theo khoang cach 2 mat.
    face_frontal_nose_tol: float = Field(default=0.18, gt=0.0, le=1.0)
    face_frontal_roll_tol: float = Field(default=0.18, gt=0.0, le=1.0)
    face_frontal_min_eye_ratio: float = Field(default=0.20, gt=0.0, le=1.0)
    # Chao khach la (unknown) cho doi dinh danh: neu trong khoang nay mat
    # duoc xac nhan thi chi chao ten, khong chao "quy khach" nua.
    unknown_greet_delay_s: float = Field(default=2.0, ge=0.0)
    # Open-palm greeting: giơ đủ bàn tay 5 ngón -> chào (trigger duy nhat).
    # Event-driven + gated: chỉ chạy ~3-5 FPS trên candidate đủ lớn.
    # Thong nhat 10s nhu cooldown loa.
    palm_every_k: int = Field(default=4, ge=1)
    palm_max_people: int = Field(default=2, ge=1)
    palm_cooldown_s: float = Field(default=10.0, ge=0)
    palm_confirm_frames: int = Field(default=1, ge=1)
    palm_release_frames: int = Field(default=2, ge=1)
    palm_min_detection_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    palm_required_fingers: int = Field(default=4, ge=3, le=5)
    palm_min_input_height_px: int = Field(default=320, ge=0)
    palm_head_region_max_y: float = Field(default=0.60, gt=0.0, le=1.0)
    palm_allow_without_face: bool = True
    palm_face_ttl_s: float = Field(default=3.0, gt=0.0)
    palm_face_max_distance: float = Field(default=2.5, gt=0.0)
    palm_min_person_area_px: float = Field(default=8000.0, ge=0)
    # Face async worker: hàng đợi job, drop cũ khi quá tải để giữ realtime.
    face_max_queue: int = Field(default=8, ge=1)
    face_max_job_age_s: float = Field(default=5.0, ge=0.5)
    face_recheck_s: float = Field(default=30.0, ge=0)
    # Voice-trigger (OR với wave): mic camera nghe cụm gọi trong
    # config/voice_triggers/ thì greet người gần nhất (unknown + employee
    # như nhau). Opt-in bằng --voice-trigger (cần --greet để có loa phát).
    # STT faster-whisper small/vi: model nhỏ ít bịa chữ từ ồn, nhẹ CPU.
    voice_trigger_enabled: bool = False
    # Folder cụm gọi (mỗi *.yaml: phrases/fillers/max_fillers). Thêm cụm
    # mới ("ok imou", ...) chỉ cần thêm file + restart, không sửa code.
    trigger_phrase_dir: Path = Path("config/voice_triggers")
    # Cụm bổ sung ngoài folder (nối thêm, ít dùng).
    voice_trigger_words: list[str] = Field(default_factory=list)
    voice_trigger_window_s: float = Field(default=5.0, ge=0)
    voice_trigger_inhibit_s: float = Field(default=12.0, ge=0)
    # P6: inhibit hiệu lực = max(cấu hình, câu chào dài nhất + đuôi vang).
    voice_trigger_echo_tail_s: float = Field(default=2.0, ge=0)
    # P5: voice chỉ hiệu lực khi có mặt tươi trong TTL này.
    voice_face_ttl_s: float = Field(default=3.0, gt=0.0)
    # P5: người được voice-greet phải đủ lớn (cùng đơn vị palm_min_person_area_px).
    voice_min_person_area_px: float = Field(default=2500.0, ge=0)
    # P1: ngưỡng VAD động bám nền (factor<=0 -> ngưỡng tĩnh như cũ).
    voice_vad_floor_factor: float = Field(default=3.0, ge=0)
    voice_vad_floor_min: float = Field(default=0.004, gt=0.0)
    voice_vad_ceiling: float = Field(default=0.15, gt=0.0)
    # P3: đoạn chỉ đi STT khi to hơn nền >= số dB này.
    voice_snr_min_db: float = Field(default=10.0, ge=0)
    voice_stt_model: str = "small"
    voice_stt_lang: str = "vi"
    voice_listen_channel: int = Field(default=1, ge=1)
    voice_listen_subtype: int = Field(default=1, ge=0)
    voice_vad_threshold: float = Field(default=0.004, ge=0)
    voice_min_seg_rms: float = Field(default=0.003, ge=0)
    # P4: model nhỏ + cửa chất lượng siết (ít bịa chữ từ ồn).
    voice_max_no_speech_prob: float = Field(default=0.40, ge=0.0, le=1.0)
    voice_min_avg_logprob: float = Field(default=-0.90, le=0.0)
    # Dump mọi đoạn voice đi STT ra WAV ("" = tắt).
    voice_dump_dir: str = "output/voice_dumps"


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
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
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
