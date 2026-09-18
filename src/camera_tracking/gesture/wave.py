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
    if not sampled:
        return 0
    # Follow turning points, not individual frame deltas. A real wave normally
    # moves through several small per-frame steps; requiring one step to cross
    # min_amplitude made ordinary, smooth waving almost impossible to trigger.
    reversals = 0
    direction = 0
    anchor = extreme = sampled[0][1]
    for _, curr_x in sampled[1:]:
        if direction == 0:
            delta = curr_x - anchor
            if abs(delta) >= min_amplitude:
                direction = 1 if delta > 0 else -1
                extreme = curr_x
            continue
        if direction > 0:
            if curr_x > extreme:
                extreme = curr_x
            elif extreme - curr_x >= min_amplitude:
                reversals += 1
                direction = -1
                extreme = curr_x
        else:
            if curr_x < extreme:
                extreme = curr_x
            elif curr_x - extreme >= min_amplitude:
                reversals += 1
                direction = 1
                extreme = curr_x
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
    """Nhan wave moi Global ID tu chuoi vi tri ngang cua ban tay.

    Tra True dung 1 lan khi du dao chieu, roi cooldown per-gid de
    khong spam voice khi vay lien tuc.
    """

    window_s: float = 1.5
    min_reversals: int = 2
    min_amplitude: float = 0.06
    min_gap_s: float = 0.08
    cooldown_s: float = 30.0
    required_fingers: float = 5
    palm_confirm_frames: int = 2
    palm_release_frames: int = 5
    max_hand_center_y: float = 0.65
    hand_detector: HandDetector | None = None
    _history: dict[int, deque] = field(default_factory=dict, init=False)
    _last_wave_s: dict[int, float] = field(default_factory=dict, init=False)
    _palm_open_counts: dict[int, int] = field(default_factory=dict, init=False)
    _palm_missing_counts: dict[int, int] = field(default_factory=dict, init=False)
    _debug_hands: dict[int, list[list[tuple[float, float]]]] = field(
        default_factory=dict, init=False)
    _debug_valid: dict[int, list[bool]] = field(default_factory=dict, init=False)
    _debug_at: dict[int, float] = field(default_factory=dict, init=False)

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

    def observe_open_palm_wave(
        self,
        gid: int,
        crop_bgr,
        now_s: float,
        *,
        detector=None,
        face_bbox: tuple[float, float, float, float] | None = None,
        face_max_distance: float = 3.5,
    ) -> bool:
        """Trigger only while a five-finger, camera-facing palm is waving.

        Unlike :meth:`observe`, this path consumes all hand landmarks and is
        the production greeting gate. Ordinary arm/wrist motion never enters
        the reversal history unless the open-palm activation is already valid.
        """
        detect = detector or self.hand_detector
        if detect is None or not hasattr(detect, "landmarks"):
            return False
        try:
            hands = detect.landmarks(crop_bgr)
        except RuntimeError:
            hands = []
        hands = list(hands or [])
        validity = [
            bool(
                is_open_palm(hand, required=self.required_fingers)
                and is_front_facing_hand(hand)
                and (sum(point[1] for point in hand) / len(hand)
                     <= max(0.1, min(1.0, self.max_hand_center_y)))
                and (face_bbox is None or is_hand_near_face(
                    hand, face_bbox, getattr(crop_bgr, "shape", ()),
                    face_max_distance))
            )
            for hand in hands
        ]
        self._debug_hands[gid] = hands
        self._debug_valid[gid] = validity
        self._debug_at[gid] = now_s
        valid = [hand for hand, accepted in zip(hands, validity) if accepted]
        if now_s - self._last_wave_s.get(gid, float("-inf")) < self.cooldown_s:
            return False
        if not valid:
            missing = self._palm_missing_counts.get(gid, 0) + 1
            self._palm_missing_counts[gid] = missing
            if missing >= max(1, self.palm_release_frames):
                # Tay ha xuong that su: xoa quy dao + reset confirm de lan
                # gio tay ke tiep phai kich hoat lai tu dau.
                self._history.pop(gid, None)
                self._palm_open_counts[gid] = 0
            # Miss ngan (motion blur / xoay tay giua nhịp vay): giu nguyen
            # history + opened de nhịp vay khong bi cat vun.
            return False

        self._palm_missing_counts[gid] = 0
        opened = self._palm_open_counts.get(gid, 0) + 1
        self._palm_open_counts[gid] = opened

        history = self._history.setdefault(gid, deque())
        cutoff = now_s - self.window_s
        while history and history[0][0] < cutoff:
            history.popleft()
        # A natural greeting often rotates the open hand around a nearly
        # stationary wrist. Tracking wrist.x alone therefore missed a visible
        # wave. The five-tip mean follows both wrist-led and whole-arm waves,
        # while averaging out one noisy landmark.
        motion_xs = [_wave_motion_x(hand) for hand in valid]
        motion_x = (min(motion_xs, key=lambda x: abs(x - history[-1][1]))
                    if history else max(motion_xs))
        history.append((now_s, motion_x))
        # Confirm chi gate viec fire, khong gate viec tich luy quy dao:
        # frame valid dau tien sau 1 miss ngan van phai vao history, neu
        # khong moi nhịp vay bi mat 1 mau va WAVE mai o 0/1.
        if opened < max(1, self.palm_confirm_frames):
            return False
        if count_reversals(
            list(history), min_gap_s=self.min_gap_s,
            min_amplitude=self.min_amplitude,
        ) < self.min_reversals:
            return False
        self._last_wave_s[gid] = now_s
        history.clear()
        self._palm_open_counts[gid] = 0
        return True

    def current_reversals(self, gid: int) -> int:
        """So dao chieu hien tai trong window (debug, khong tac dung phu)."""
        points = list(self._history.get(gid, ()))
        return count_reversals(
            points,
            min_gap_s=self.min_gap_s,
            min_amplitude=self.min_amplitude,
        )

    def debug_snapshot(
        self, gid: int, now_s: float, max_age_s: float = 1.0,
    ) -> tuple[list[list[tuple[float, float]]], list[bool]] | None:
        """Return recent MediaPipe points for a read-only live overlay."""
        if now_s - self._debug_at.get(gid, float("-inf")) > max_age_s:
            return None
        return (self._debug_hands.get(gid, []),
                self._debug_valid.get(gid, []))

    def forget_retired(self, alive_gids: set[int]) -> None:
        for gid in [g for g in self._history if g not in alive_gids]:
            self._history.pop(gid, None)
        for gid in [g for g in self._last_wave_s if g not in alive_gids]:
            self._last_wave_s.pop(gid, None)
        for state in (
            self._palm_open_counts, self._palm_missing_counts,
            self._debug_hands, self._debug_valid, self._debug_at,
        ):
            for gid in [g for g in state if g not in alive_gids]:
                state.pop(gid, None)

    def remap_gid(self, old_gid: int, new_gid: int) -> None:
        if old_gid == new_gid:
            return
        old_history = self._history.pop(old_gid, None)
        if old_history:
            merged = list(self._history.get(new_gid, ())) + list(old_history)
            self._history[new_gid] = deque(sorted(merged, key=lambda row: row[0]))
        for state in (
            self._last_wave_s, self._palm_open_counts,
            self._palm_missing_counts, self._debug_at,
        ):
            if old_gid in state:
                state[new_gid] = max(state.get(new_gid, state[old_gid]),
                                     state.pop(old_gid))
        for state in (self._debug_hands, self._debug_valid):
            if old_gid in state:
                state[new_gid] = state.pop(old_gid)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _wave_motion_x(landmarks: list[tuple[float, float]]) -> float:
    """Horizontal open-hand position used for wave direction changes."""
    if len(landmarks) < 21:
        return float(landmarks[0][0]) if landmarks else 0.0
    return sum(float(landmarks[index][0])
               for index in (4, 8, 12, 16, 20)) / 5.0


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


def _clamp01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def open_palm_score(landmarks: list[tuple[float, float]]) -> float:
    """Diem mo ban tay 0..1 (trung binh diem duoi tung ngon / 5).

    Moi ngon cham lien tuc thay vi nhi phan: ti le khoang cach
    tip/pip = 1.0 (gap) -> 0 diem, >= nguong cu (1.08 ngon dai / 1.03
    ngon cai) -> 1 diem. Ngon gan-du-nguong van duoc diem mot phan
    thay vi 0 nhu truoc, chong truong hop thieu 1 chut la rot.
    """
    if landmarks is None or len(landmarks) < 21:
        return 0.0
    try:
        wrist = landmarks[0]
        total = 0.0
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            denom = _dist(landmarks[pip], wrist)
            if denom <= 1e-9:
                continue
            ratio = _dist(landmarks[tip], wrist) / denom
            total += _clamp01((ratio - 1.0) / 0.08)
        denom = _dist(landmarks[3], wrist)
        if denom > 1e-9:
            ratio = _dist(landmarks[4], wrist) / denom
            base = _clamp01((ratio - 1.0) / 0.03)
            sep = abs(float(landmarks[4][0]) - float(landmarks[5][0]))
            total += base * _clamp01((sep - 0.02) / 0.02)
        return _clamp01(total / 5.0)
    except (IndexError, TypeError, ValueError):
        return 0.0


def is_open_palm(
    landmarks: list[tuple[float, float]],
    required: float = 5,
) -> bool:
    """True khi diem mo ban tay dat nguong (mac dinh 5/5 = 100%).

    required la so ngon-tuong-duong (float): 4.5 = 90%, cho phep 1 ngon
    gan-du (VD kheo tay hoi cong) van qua. Dung hinh hoc tuong doi voi
    co tay nen ban tay van hop le khi nghieng trong luc vay.
    """
    if landmarks is None or len(landmarks) < 21:
        return False
    try:
        need = max(0.0, min(5.0, float(required)))
    except (TypeError, ValueError):
        return False
    return bool(open_palm_score(landmarks) * 5.0 >= need)


def is_front_facing_hand(
    landmarks: list[tuple[float, float]],
    min_palm_width_ratio: float = 0.20,
) -> bool:
    """Reject an edge-on/foreshortened hand using visible palm geometry.

    MediaPipe's 2-D points cannot reliably distinguish palm from the back of
    the hand, but palm width versus wrist-to-middle-finger length robustly
    rejects the side-on poses that commonly produce accidental activation.
    """
    if landmarks is None or len(landmarks) < 21:
        return False
    try:
        palm_width = _dist(landmarks[5], landmarks[17])
        hand_length = _dist(landmarks[0], landmarks[12])
        return hand_length > 1e-6 and palm_width / hand_length >= max(
            0.05, min_palm_width_ratio)
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
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

    required_fingers: float = 5
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

    def remap_gid(self, old_gid: int, new_gid: int) -> None:
        if old_gid == new_gid:
            return
        for state in (self._last_fire_s, self._open_counts, self._closed_counts):
            if old_gid in state:
                state[new_gid] = max(state.get(new_gid, state[old_gid]),
                                     state.pop(old_gid))
        if old_gid in self._latched:
            self._latched.discard(old_gid)
            self._latched.add(new_gid)


__all__ = [
    "HandDetector",
    "MediaPipeHandDetector",
    "OpenPalmDetector",
    "WaveDetector",
    "count_extended_fingers",
    "open_palm_score",
    "count_reversals",
    "is_front_facing_hand",
    "is_hand_near_face",
    "is_open_palm",
]
