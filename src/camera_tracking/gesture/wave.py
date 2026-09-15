"""Phat hien vay tay (wave) bang MediaPipe Hands + dao chieu co tay.

Thiet ke de test offline: hand detector inject duoc (mac dinh MediaPipe),
logic dem dao chieu la ham thuan tuy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Protocol


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
    """HandDetector that, lay wrist.x lon nhat moi ban tay."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self.max_num_hands = max(1, max_num_hands)
        self.min_detection_confidence = min_detection_confidence
        self.static_image_mode = static_image_mode
        self._hands = None

    def _load(self):
        if self._hands is None:
            try:
                import mediapipe as mp
            except ImportError as error:
                raise RuntimeError(
                    "mediapipe chua cai. Chay: pip install mediapipe"
                ) from error
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
            )
        return self._hands

    @property
    def available(self) -> bool:
        try:
            self._load()
            return True
        except RuntimeError:
            return False

    def __call__(self, crop_bgr) -> list[float] | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            import cv2
        except ImportError:
            return None
        hands = self._load()
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        return [hand.landmark[0].x for hand in result.multi_hand_landmarks]


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

    def forget_retired(self, alive_gids: set[int]) -> None:
        for gid in [g for g in self._history if g not in alive_gids]:
            self._history.pop(gid, None)
        for gid in [g for g in self._last_wave_s if g not in alive_gids]:
            self._last_wave_s.pop(gid, None)


__all__ = ["HandDetector", "MediaPipeHandDetector", "WaveDetector", "count_reversals"]
