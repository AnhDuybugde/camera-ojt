"""Wave detector: dem dao chieu co tay (stub hand, khong can camera)."""
import sys
import types
from types import SimpleNamespace

import numpy as np

from camera_tracking.gesture.wave import (
    MediaPipeHandDetector,
    OpenPalmDetector,
    WaveDetector,
    count_extended_fingers,
    count_reversals,
    is_hand_near_face,
    is_open_palm,
)


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


def test_current_reversals_has_no_side_effects() -> None:
    detector = WaveDetector(min_reversals=100)
    detect = _stub_hands([[0.4]])
    detector.observe(7, None, 0.0, detector=detect)
    assert detector.current_reversals(7) == 0
    assert detector.current_reversals(999) == 0
    # Doc khong xoa lich su, khong dat cooldown.
    assert 7 in detector._history
    assert 7 not in detector._last_wave_s


def test_forget_retired() -> None:
    detector = WaveDetector(min_reversals=100)
    detect = _stub_hands([[0.4]])
    detector.observe(7, None, 0.0, detector=detect)
    assert 7 in detector._history
    detector.forget_retired(set())
    assert 7 not in detector._history


def _stub_mediapipe_tasks(monkeypatch, tmp_path, hands_per_call):
    """Gia lap mediapipe>=1.0 (chi co mp.tasks, khong co mp.solutions)."""
    state = {"options": None, "timestamps": [], "calls": 0}

    mp_module = types.ModuleType("mediapipe")
    tasks_module = types.ModuleType("mediapipe.tasks")
    python_module = types.ModuleType("mediapipe.tasks.python")
    vision_module = types.ModuleType("mediapipe.tasks.python.vision")

    class FakeImage:
        def __init__(self, image_format, data):
            self.image_format = image_format
            self.data = data

    class FakeRunningMode:
        VIDEO = "VIDEO"

    def fake_base_options(*, model_asset_path):
        return SimpleNamespace(model_asset_path=model_asset_path)

    def fake_options(**kwargs):
        state["options"] = kwargs
        return SimpleNamespace(**kwargs)

    class FakeLandmarker:
        @staticmethod
        def create_from_options(options):
            state["options"] = options
            return FakeLandmarker()

        def detect_for_video(self, image, timestamp_ms):
            state["timestamps"].append(timestamp_ms)
            idx = state["calls"]
            state["calls"] += 1
            hands = hands_per_call[min(idx, len(hands_per_call) - 1)]
            if hands is None:
                return SimpleNamespace(hand_landmarks=[])
            return SimpleNamespace(
                hand_landmarks=[[SimpleNamespace(x=x)] for x in hands]
            )

    mp_module.Image = FakeImage
    mp_module.ImageFormat = SimpleNamespace(SRGB="SRGB")
    mp_module.tasks = tasks_module
    tasks_module.python = python_module
    python_module.BaseOptions = fake_base_options
    python_module.vision = vision_module
    vision_module.HandLandmarkerOptions = fake_options
    vision_module.RunningMode = FakeRunningMode
    vision_module.HandLandmarker = FakeLandmarker

    model_file = tmp_path / "hand_landmarker.task"
    model_file.write_bytes(b"fake-model")
    for name, module in (
        ("mediapipe", mp_module),
        ("mediapipe.tasks", tasks_module),
        ("mediapipe.tasks.python", python_module),
        ("mediapipe.tasks.python.vision", vision_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return state, model_file


def test_tasks_detector_returns_wrist_x(monkeypatch, tmp_path) -> None:
    state, model_file = _stub_mediapipe_tasks(
        monkeypatch, tmp_path, [[0.3, 0.7], None]
    )
    detector = MediaPipeHandDetector(
        max_num_hands=2, min_detection_confidence=0.6,
        model_path=model_file,
    )
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    assert detector(crop) == [0.3, 0.7]
    assert detector(crop) is None
    # Tham so constructor duoc truyen xuong Tasks options; timestamp VIDEO
    # tang don dieu moi lan goi.
    assert state["options"].num_hands == 2
    assert state["options"].min_hand_detection_confidence == 0.6
    assert state["timestamps"] == [1, 2]


def test_tasks_detector_empty_crop_skips_model(monkeypatch, tmp_path) -> None:
    state, model_file = _stub_mediapipe_tasks(monkeypatch, tmp_path, [[0.5]])
    detector = MediaPipeHandDetector(model_path=model_file)
    assert detector(np.zeros((0, 0, 3), dtype=np.uint8)) is None
    assert detector._hands is None
    assert state["calls"] == 0


def _palm_landmarks(open_hand: bool = True) -> list:
    """21 landmarks giả: xoè (tips xa wrist) vs nắm (tips gần wrist)."""
    pts = [(0.5, 0.5)] * 21
    pts[0] = (0.5, 0.9)  # wrist
    if open_hand:
        tips_y, pips_y = 0.15, 0.5
        pts[3] = (0.35, 0.6)
        pts[4] = (0.25, 0.5)
    else:
        tips_y, pips_y = 0.7, 0.5
        pts[3] = (0.45, 0.7)
        pts[4] = (0.5, 0.8)
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        x = 0.4 + (tip % 5) * 0.05
        pts[pip] = (x, pips_y)
        pts[tip] = (x, tips_y)
    return pts


def test_count_extended_fingers_open_palm() -> None:
    assert count_extended_fingers(_palm_landmarks(True)) == 5


def test_count_extended_fingers_fist() -> None:
    assert count_extended_fingers(_palm_landmarks(False)) == 0


def test_count_extended_fingers_short_input() -> None:
    assert count_extended_fingers([(0.1, 0.1)] * 5) == 0
    assert is_open_palm(_palm_landmarks(True)) is True
    assert is_open_palm(_palm_landmarks(False)) is False


def test_open_palm_rejects_extended_but_downward_fingers() -> None:
    hand = _palm_landmarks(True)
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        hand[tip] = (hand[tip][0], hand[pip][1] + 0.2)
    assert is_open_palm(hand) is False


def test_open_palm_required_four_allows_occluded_thumb() -> None:
    hand = _palm_landmarks(True)
    hand[4] = hand[3]
    assert is_open_palm(hand, required=4) is True
    assert is_open_palm(hand, required=5) is False


def test_hand_must_be_near_face_in_same_person_crop() -> None:
    hand = _palm_landmarks(True)
    assert is_hand_near_face(hand, (25, 20, 75, 70), (200, 100, 3), 2.5)
    assert not is_hand_near_face(hand, (0, 0, 10, 10), (1000, 1000, 3), 2.5)


class _StubLandmarkDetector:
    def __init__(self, hands):
        self._hands = hands

    def landmarks(self, crop):
        return self._hands

    def __call__(self, crop):
        return [0.5] if self._hands else None


def test_open_palm_requires_confirmation_and_release_before_refire() -> None:
    detector = OpenPalmDetector(
        required_fingers=5, cooldown_s=30.0,
        confirm_frames=2, release_frames=2,
    )
    stub = _StubLandmarkDetector([_palm_landmarks(True)])
    assert detector.observe(3, None, 0.0, detector=stub) is False
    assert detector.observe(3, None, 0.5, detector=stub) is True
    # Giữ tay qua cả cooldown vẫn không được chào lại.
    assert detector.observe(3, None, 31.0, detector=stub) is False
    closed = _StubLandmarkDetector([])
    assert detector.observe(3, None, 32.0, detector=closed) is False
    assert detector.observe(3, None, 32.5, detector=closed) is False
    assert detector.observe(3, None, 33.0, detector=stub) is False
    assert detector.observe(3, None, 33.5, detector=stub) is True

    assert detector.observe(4, None, 1.0, detector=stub) is False
    assert detector.observe(4, None, 1.5, detector=stub) is True


def test_open_palm_ignores_fist_and_empty() -> None:
    detector = OpenPalmDetector(required_fingers=5, cooldown_s=0.0)
    assert detector.observe(5, None, 0.0,
                            detector=_StubLandmarkDetector(
                                [_palm_landmarks(False)])) is False
    assert detector.observe(6, None, 0.0,
                            detector=_StubLandmarkDetector([])) is False


def test_open_palm_rejects_hand_below_head_region_without_face() -> None:
    hand = _palm_landmarks(True)
    hand = [(x, min(0.99, y + 0.35)) for x, y in hand]
    detector = OpenPalmDetector(
        required_fingers=4, cooldown_s=0.0, confirm_frames=1,
        max_hand_center_y=0.60,
    )
    assert detector.observe(
        8, None, 0.0, detector=_StubLandmarkDetector([hand]),
    ) is False


def test_open_palm_forget_retired() -> None:
    detector = OpenPalmDetector(cooldown_s=30.0, confirm_frames=1)
    stub = _StubLandmarkDetector([_palm_landmarks(True)])
    assert detector.observe(7, None, 0.0, detector=stub) is True
    detector.forget_retired(set())
    assert 7 not in detector._last_fire_s
    assert 7 not in detector._latched
