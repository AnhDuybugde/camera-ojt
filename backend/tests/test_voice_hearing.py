"""P1+P3 hearing: ngưỡng VAD động bám nền + cổng SNR đoạn."""
from camera_tracking.voice.rtsp_voice_listener import (
    NoiseFloorTracker,
    db_to_amplitude_ratio,
)


def test_db_ratio_10db_is_about_316() -> None:
    assert abs(db_to_amplitude_ratio(10.0) - 3.1623) < 0.01
    assert db_to_amplitude_ratio(0.0) == 1.0


def test_noise_floor_starts_at_min_and_adapts_up() -> None:
    tracker = NoiseFloorTracker(
        static_threshold=0.012, factor=3.0, abs_min=0.02, ceiling=0.15)
    # Mới khởi động: ngưỡng = sàn tuyệt đối x factor.
    assert tracker.threshold == 0.06
    # Phòng ồn đều 0.03: nền học lên, ngưỡng bám theo.
    for _ in range(300):
        tracker.update(0.03)
    assert abs(tracker.floor - 0.03) < 0.005
    assert abs(tracker.threshold - 0.09) < 0.015


def test_noise_floor_ignores_short_loud_burst() -> None:
    tracker = NoiseFloorTracker(
        static_threshold=0.012, factor=3.0, abs_min=0.02, ceiling=0.15)
    for _ in range(100):
        tracker.update(0.01)
    floor_before = tracker.floor
    # Tiếng nói ngắn (1 frame to) gần như không kéo nền lên.
    tracker.update(0.20)
    assert abs(tracker.floor - floor_before) < 0.001


def test_noise_floor_respects_ceiling() -> None:
    tracker = NoiseFloorTracker(
        static_threshold=0.012, factor=3.0, abs_min=0.02, ceiling=0.10)
    for _ in range(500):
        tracker.update(0.09)
    assert tracker.threshold == 0.10


def test_noise_floor_static_mode_when_factor_disabled() -> None:
    tracker = NoiseFloorTracker(static_threshold=0.05, factor=0.0)
    assert tracker.threshold == 0.05
    tracker.update(0.01)
    assert tracker.threshold == 0.05


def test_snr_gate_logic_matches_listener() -> None:
    """Đoạn to hơn nền >= 10dB mới đi STT (P3)."""
    import math

    ratio = db_to_amplitude_ratio(10.0)
    floor = 0.02
    assert 0.10 >= floor * ratio  # tiếng gọi gần -> qua
    assert not (0.05 >= floor * ratio)  # tiếng xa/ồn nhỏ -> bỏ
    assert math.isclose(20 * math.log10(ratio), 10.0, abs_tol=0.01)
