"""Wave detector: dem dao chieu co tay (stub hand, khong can camera)."""
from camera_tracking.gesture.wave import WaveDetector, count_reversals


def _wave_points(start: float = 0.0, step: float = 0.1,
                 xs=(0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3)) -> list:
    return [(start + i * step, x) for i, x in enumerate(xs)]


def test_count_reversals_detects_wave() -> None:
    # 7 diem -> 6 doan, doan dau dat huong, 5 doan sau doi huong.
    assert count_reversals(_wave_points()) == 5


def test_count_reversals_ignores_jitter() -> None:
    still = [(i * 0.1, 0.40 + (0.01 if i % 2 else -0.01)) for i in range(10)]
    assert count_reversals(still) == 0


def test_count_reversals_ignores_slow_drift() -> None:
    drift = [(i * 0.5, 0.3 + i * 0.02) for i in range(6)]
    assert count_reversals(drift) == 0


def _stub_hands(frames: list):
    state = {"i": 0}

    def detect(crop):
        idx = state["i"]
        state["i"] += 1
        return frames[min(idx, len(frames) - 1)]

    return detect


def test_wave_fires_once_then_cooldown() -> None:
    xs_seq = [[x] for x in (0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3)]
    detector = WaveDetector(min_reversals=4, cooldown_s=30.0)
    detect = _stub_hands(xs_seq)
    fired = [detector.observe(7, None, i * 0.1, detector=detect)
             for i in range(len(xs_seq))]
    assert fired.count(True) == 1
    # Vay tiep trong cooldown -> khong fire.
    assert detector.observe(7, None, 1.0, detector=detect) is False


def test_no_hands_no_wave() -> None:
    detector = WaveDetector(min_reversals=4)
    detect = _stub_hands([None, None, None])
    assert all(detector.observe(9, None, i * 0.1, detector=detect) is False
               for i in range(3))


def test_forget_retired() -> None:
    detector = WaveDetector(min_reversals=100)
    detect = _stub_hands([[0.4]])
    detector.observe(7, None, 0.0, detector=detect)
    assert 7 in detector._history
    detector.forget_retired(set())
    assert 7 not in detector._history
