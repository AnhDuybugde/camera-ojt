"""Phat hien vay tay (wave) bang MediaPipe Hands + dao chieu co tay.

Thiet ke de test offline: hand detector inject duoc (mac dinh MediaPipe),
logic dem dao chieu la ham thuan tuy.

Tuong thich ca 2 the he MediaPipe: Tasks HandLandmarker (mediapipe>=0.10,
bat buoc tu 1.0 vi `mp.solutions` da bi xoa) va legacy `mp.solutions.hands`.
"""
from __future__ import annotations

import shutil
from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Protocol

_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_HAND_LANDMARKER_PATH = Path("models/hand_landmarker.task")


def _ensure_landmarker_model(path: str | Path | None) -> Path:
    """Tai hand_landmarker.task lan dau (cache), cac lan sau offline."""
    dest = Path(path) if path else _HAND_LANDMARKER_PATH
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        from urllib.request import urlopen

        with urlopen(_HAND_LANDMARKER_URL, timeout=120) as src, \
                open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 256)
    except Exception as error:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"Khong tai duoc hand_landmarker.task ({error}); "
            "kiem tra mang roi chay lai."
        ) from error
    tmp.replace(dest)
    return dest


class HandDetector(Protocol):
    """Tra xs chuan hoa 0..1 cua co tay cac ban tay thay duoc (None = khong thay)."""

    def __call__(self, crop_bgr) -> list[float] | None: ...


def count_reversals(
    points: list[tuple[float, float]],
    *,
    min_gap_s: float = 0.08,
    min_amplitude: float = 0.06,
) -> int:
    """Dem so lan doi huong trai/phai voi bien do du lon.

    points: [(timestamp_s, wrist_x_norm)] tang dan theo thoi gian.
    """
    # Giu 1 diem dai dien moi khoang min_gap_s de loc rung tay.
    sampled: list[tuple[float, float]] = []
    for stamp, x in points:
        if sampled and stamp - sampled[-1][0] < min_gap_s:
            sampled[-1] = (stamp, x)
        else:
            sampled.append((stamp, x))
    reversals = 0
    direction = 0
    for (_, prev_x), (_, curr_x) in pairwise(sampled):
        delta = curr_x - prev_x
        if abs(delta) < min_amplitude:
            continue
        sign = 1 if delta > 0 else -1
        if direction != 0 and sign != direction:
            reversals += 1
        direction = sign
    return reversals


class MediaPipeHandDetector:
    """HandDetector that, lay wrist.x moi ban tay (Tasks API, fallback legacy)."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        static_image_mode: bool = False,
        model_path: str | Path | None = None,
        min_input_height_px: int = 0,
    ) -> None:
        self.max_num_hands = max(1, max_num_hands)
        self.min_detection_confidence = min_detection_confidence
        self.static_image_mode = static_image_mode
        self.model_path = model_path
        self.min_input_height_px = max(0, int(min_input_height_px))
        self._hands = None
        self._tasks_backend = False
        self._tick_ms = 0

    def _load(self):
        if self._hands is not None:
            return self._hands
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "mediapipe chua cai. Chay: pip install mediapipe"
            ) from error
        if hasattr(mp, "tasks"):
            self._hands = self._load_tasks_bound(mp)
            self._tasks_backend = True
        elif hasattr(mp, "solutions"):
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
            )
        else:
            raise RuntimeError(
                "mediapipe khong co tasks lan solutions; "
                "nang cap: pip install -U mediapipe"
            )
        return self._hands

    def _load_tasks_bound(self, mp):  # type: ignore[no-untyped-def]
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_file = _ensure_landmarker_model(self.model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_file)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_detection_confidence,
        )
        return mp_vision.HandLandmarker.create_from_options(options)

    @property
    def available(self) -> bool:
        try:
            import mediapipe as mp
        except ImportError:
            return False
        return hasattr(mp, "tasks") or hasattr(mp, "solutions")

    def ensure_loaded(self) -> None:
        """Tai model + khoi tao landmarker ngay (loi ro rang de tat voice)."""
        if self._hands is not None:
            return
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "mediapipe chua cai. Chay: pip install mediapipe"
            ) from error
        if hasattr(mp, "tasks"):
            self._hands = self._load_tasks_bound(mp)
            self._tasks_backend = True
        elif hasattr(mp, "solutions"):
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
            )
        else:
            raise RuntimeError(
                "mediapipe khong co tasks lan solutions; "
                "nang cap: pip install -U mediapipe"
            )

    def __call__(self, crop_bgr) -> list[float] | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            import cv2
        except ImportError:
            return None
        hands = self._load()
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if self._tasks_backend:
            import mediapipe as mp

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._tick_ms += 1
            result = hands.detect_for_video(image, self._tick_ms)
            if not result.hand_landmarks:
                return None
            return [hand[0].x for hand in result.hand_landmarks]
        result = hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        return [hand.landmark[0].x for hand in result.multi_hand_landmarks]

    def landmarks(self, crop_bgr) -> list[list[tuple[float, float]]]:
        """Trả full 21 landmarks mỗi bàn tay (chuẩn hoá 0..1).

        Dùng cho open-palm 4-5 ngón. Trả [] khi không thấy tay.
        """
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return []
        try:
            import cv2
        except ImportError:
            return []
        hands = self._load()
        crop_h = int(crop_bgr.shape[0])
        if 0 < crop_h < self.min_input_height_px:
            scale = self.min_input_height_px / max(1, crop_h)
            crop_bgr = cv2.resize(
                crop_bgr, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if self._tasks_backend:
            import mediapipe as mp

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._tick_ms += 1
            result = hands.detect_for_video(image, self._tick_ms)
            if not result.hand_landmarks:
                return []
            return [[(float(p.x), float(p.y)) for p in hand]
                    for hand in result.hand_landmarks]
        result = hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []
        return [[(float(p.x), float(p.y)) for p in hand.landmark]
                for hand in result.multi_hand_landmarks]


@dataclass
class WaveDetector:
    """Nhan wave moi Global ID tu chuoi wrist-x theo thoi gian.

    Tra True dung 1 lan khi du dao chieu, roi cooldown per-gid de
    khong spam voice khi vay lien tuc.
    """

    window_s: float = 1.5
    min_reversals: int = 4
    min_amplitude: float = 0.06
    min_gap_s: float = 0.08
    cooldown_s: float = 30.0
    hand_detector: HandDetector | None = None
    _history: dict[int, deque] = field(default_factory=dict, init=False)
    _last_wave_s: dict[int, float] = field(default_factory=dict, init=False)

    def observe(
        self,
        gid: int,
        crop_bgr,
        now_s: float,
        detector: HandDetector | None = None,
    ) -> bool:
        """Cap nhat 1 frame crop; True khi vua hoan thanh 1 cai vay tay."""
        detect = detector or self.hand_detector
        if detect is None:
            return False
        if now_s - self._last_wave_s.get(gid, float("-inf")) < self.cooldown_s:
            return False
        try:
            wrists = detect(crop_bgr)
        except RuntimeError:
            return False
        history = self._history.setdefault(gid, deque())
        cutoff = now_s - self.window_s
        while history and history[0][0] < cutoff:
            history.popleft()
        if wrists:
            # Vay 1 tay: lay bien do lon nhat; tranh nhay giua 2 tay bang
            # cach bam tay gan nhat voi diem cuoi.
            if history:
                last_x = history[-1][1]
                best = min(wrists, key=lambda x: abs(x - last_x))
            else:
                best = max(wrists)
            history.append((now_s, best))
        points = list(history)
        if count_reversals(
            points,
            min_gap_s=self.min_gap_s,
            min_amplitude=self.min_amplitude,
        ) >= self.min_reversals:
            self._last_wave_s[gid] = now_s
            history.clear()
            return True
        return False

    def current_reversals(self, gid: int) -> int:
        """So dao chieu hien tai trong window (debug, khong tac dung phu)."""
        points = list(self._history.get(gid, ()))
        return count_reversals(
            points,
            min_gap_s=self.min_gap_s,
            min_amplitude=self.min_amplitude,
        )

    def forget_retired(self, alive_gids: set[int]) -> None:
        for gid in [g for g in self._history if g not in alive_gids]:
            self._history.pop(gid, None)
        for gid in [g for g in self._last_wave_s if g not in alive_gids]:
            self._last_wave_s.pop(gid, None)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def count_extended_fingers(landmarks: list[tuple[float, float]]) -> int:
    """Đếm ngón duỗi từ 21 landmarks MediaPipe (chuẩn hoá 0..1).

    Heuristic khoảng cách: đầu ngón xa cổ tay hơn khớp giữa -> duỗi.
    Đơn giản, đủ cho case "giơ bàn tay mở 4-5 ngón để chào".
    Trả 0..5.
    """
    if landmarks is None or len(landmarks) < 21:
        return 0
    try:
        wrist = landmarks[0]
        # (tip, pip) cho 4 ngón dài + (tip, ip) cho ngón cái.
        pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        count = 0
        for tip, pip in pairs:
            if _dist(landmarks[tip], wrist) > _dist(landmarks[pip], wrist) * 1.12:
                count += 1
        # Ngón cái: so tip(4) với ip(3), ngưỡng thấp hơn vì chuyển động ngang.
        if _dist(landmarks[4], wrist) > _dist(landmarks[3], wrist) * 1.05:
            count += 1
        return count
    except (IndexError, TypeError, ValueError):
        return 0


def is_open_palm(landmarks: list[tuple[float, float]], required: int = 5) -> bool:
    """True khi thấy bàn tay dựng, xoè đủ ngón.

    Ngoài khoảng cách tới cổ tay, yêu cầu bốn đầu ngón nằm phía trên khớp
    PIP và ngón cái mở sang bên. Điều kiện hướng này loại phần lớn dương tính
    giả từ mặt/quần áo trong person crop; cử chỉ chào của camera là tay dựng.
    """
    if landmarks is None or len(landmarks) < 21:
        return False
    try:
        wrist = landmarks[0]
        long_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
        # Dem tung ngon dai theo ca khoang cach va huong. Khong bat buoc ca
        # bon ngon dai khi cau hinh required=4: mot ngon bi che van hop le.
        long_extended = sum(
            _dist(landmarks[tip], wrist) > _dist(landmarks[pip], wrist) * 1.08
            and landmarks[tip][1] < landmarks[pip][1] - 0.015
            for tip, pip in long_pairs
        )
        thumb_extended = (
            _dist(landmarks[4], wrist) > _dist(landmarks[3], wrist) * 1.03
            and abs(landmarks[4][0] - landmarks[5][0]) >= 0.04
        )
        if long_extended + int(thumb_extended) < max(1, required):
            return False
        # Ít nhất ba ngón dài phải hướng lên để tránh nhầm cánh tay/vật thể.
        if long_extended < min(3, max(1, required)):
            return False
        # Cổ tay phải thấp hơn các khớp gốc: bàn tay thực sự đang dựng lên.
        wrist_y = landmarks[0][1]
        if wrist_y <= max(landmarks[i][1] for i in (5, 9, 13, 17)) + 0.02:
            return False
        return True
    except (IndexError, TypeError, ValueError):
        return False


def is_hand_near_face(
    landmarks: list[tuple[float, float]],
    face_bbox: tuple[float, float, float, float],
    crop_shape: tuple[int, ...],
    max_distance: float = 2.5,
) -> bool:
    """True khi tâm bàn tay ở gần mặt trong cùng một person crop.

    ``landmarks`` là toạ độ chuẩn hoá của MediaPipe, còn ``face_bbox`` là
    pixel tương đối theo crop người. Khoảng cách được chuẩn hoá theo kích
    thước khuôn mặt để hoạt động giống nhau khi người đứng gần hoặc xa.
    """
    if len(landmarks) < 21 or len(crop_shape) < 2:
        return False
    try:
        crop_h, crop_w = float(crop_shape[0]), float(crop_shape[1])
        fx1, fy1, fx2, fy2 = (float(v) for v in face_bbox)
        face_size = max(1.0, fx2 - fx1, fy2 - fy1)
        face_cx, face_cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        hand_cx = sum(point[0] for point in landmarks) / len(landmarks) * crop_w
        hand_cy = sum(point[1] for point in landmarks) / len(landmarks) * crop_h
        distance = _dist((hand_cx, hand_cy), (face_cx, face_cy)) / face_size
        # Tay chào có thể ở hai bên mặt, nhưng không được nằm sâu dưới thân.
        return distance <= max(0.1, max_distance) and hand_cy <= fy2 + face_size
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        return False


@dataclass
class OpenPalmDetector:
    """Greeting trigger đơn giản: giơ bàn tay mở 4-5 ngón -> True 1 lần.

    Không cần vẫy qua lại như WaveDetector cũ. Event-driven + gated ở
    caller (chỉ gọi ~3-5 FPS trên candidate đủ lớn), bên trong chỉ check
    nhiều frame liên tiếp + cooldown per-gid. Sau khi fire, người dùng phải
    hạ tay đủ số frame release rồi giơ lại; giữ nguyên bàn tay không lặp greet.
    """

    required_fingers: int = 5
    cooldown_s: float = 30.0
    confirm_frames: int = 4
    release_frames: int = 2
    hand_detector: HandDetector | None = None
    max_hand_center_y: float = 0.60
    _last_fire_s: dict[int, float] = field(default_factory=dict, init=False)
    _open_counts: dict[int, int] = field(default_factory=dict, init=False)
    _closed_counts: dict[int, int] = field(default_factory=dict, init=False)
    _latched: set[int] = field(default_factory=set, init=False)

    def observe(
        self,
        gid: int,
        crop_bgr,
        now_s: float,
        detector: HandDetector | None = None,
        face_bbox: tuple[float, float, float, float] | None = None,
        face_max_distance: float = 2.5,
    ) -> bool:
        """True một lần cho mỗi động tác hạ tay rồi giơ bàn tay xoè."""
        detect = detector or self.hand_detector
        if detect is None:
            return False
        try:
            if hasattr(detect, "landmarks"):
                hands = detect.landmarks(crop_bgr)  # type: ignore[attr-defined]
            else:
                return False
        except RuntimeError:
            return False
        palm_open = any(
            is_open_palm(hand, self.required_fingers)
            and (sum(point[1] for point in hand) / len(hand)
                 <= max(0.1, min(1.0, self.max_hand_center_y)))
            and (face_bbox is None or is_hand_near_face(
                hand, face_bbox, getattr(crop_bgr, "shape", ()), face_max_distance))
            for hand in (hands or [])
        )
        if not palm_open:
            self._open_counts[gid] = 0
            closed = self._closed_counts.get(gid, 0) + 1
            self._closed_counts[gid] = closed
            if closed >= max(1, self.release_frames):
                self._latched.discard(gid)
            return False

        self._closed_counts[gid] = 0
        opened = self._open_counts.get(gid, 0) + 1
        self._open_counts[gid] = opened
        if gid in self._latched or opened < max(1, self.confirm_frames):
            return False
        # Latched even during cooldown: holding a newly raised palm until the
        # cooldown expires must not cause a delayed greeting.
        self._latched.add(gid)
        if now_s - self._last_fire_s.get(gid, float("-inf")) < self.cooldown_s:
            return False
        self._last_fire_s[gid] = now_s
        return True

    def cooldown_remaining(self, gid: int, now_s: float) -> float:
        """So giay con lai truoc khi gid duoc fire tiep (0 = san sang)."""
        return max(0.0, self.cooldown_s - (now_s - self._last_fire_s.get(gid, float("-inf"))))

    def forget_retired(self, alive_gids: set[int]) -> None:
        for gid in [g for g in self._last_fire_s if g not in alive_gids]:
            self._last_fire_s.pop(gid, None)
        for gid in [g for g in self._open_counts if g not in alive_gids]:
            self._open_counts.pop(gid, None)
        for gid in [g for g in self._closed_counts if g not in alive_gids]:
            self._closed_counts.pop(gid, None)
        self._latched.intersection_update(alive_gids)


__all__ = [
    "HandDetector",
    "MediaPipeHandDetector",
    "OpenPalmDetector",
    "WaveDetector",
    "count_extended_fingers",
    "count_reversals",
    "is_hand_near_face",
    "is_open_palm",
]
