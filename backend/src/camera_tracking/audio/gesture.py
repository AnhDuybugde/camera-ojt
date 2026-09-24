"""Optional asynchronous open-palm gesture detector for Bé Xinh.

MediaPipe Hands is intentionally isolated from the main tracking loop. Camera
AIM submits at most a couple of small person crops every ~300 ms to one
background worker. If MediaPipe is not installed, the rest of Camera AIM keeps
working and gesture interaction is simply disabled.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import base64
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Iterable

import cv2
import numpy as np

from camera_tracking.domain import Track


@dataclass(frozen=True)
class GestureEvent:
    kind: str
    channel: str
    global_id: int
    now_s: float
    confidence: float


@dataclass(frozen=True)
class _Candidate:
    channel: str
    global_id: int
    crop: np.ndarray


@dataclass(frozen=True)
class _RawPalm:
    channel: str
    global_id: int
    confidence: float
    now_s: float


@dataclass(frozen=True)
class _Landmark:
    x: float
    y: float


class _MediaPipeSidecar:
    """Synchronous client called only from the gesture background thread."""

    def __init__(self, command: list[str]) -> None:
        worker = Path(__file__).with_name("gesture_worker.py")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            [*command, str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=creationflags,
        )
        self._request_id = 0
        assert self._process.stdout is not None
        ready_line = self._process.stdout.readline()
        if not ready_line or not json.loads(ready_line).get("ready"):
            raise RuntimeError("MediaPipe sidecar did not start")

    def process(self, image: np.ndarray) -> list[list[_Landmark]]:
        process = self._process
        if process.poll() is not None:
            raise RuntimeError("MediaPipe sidecar stopped unexpectedly")
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return []
        self._request_id += 1
        request_id = self._request_id
        payload = {
            "id": request_id,
            "image": base64.b64encode(jpeg).decode("ascii"),
        }
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        process.stdin.flush()
        response_line = process.stdout.readline()
        if not response_line:
            raise RuntimeError("MediaPipe sidecar returned no result")
        response = json.loads(response_line)
        if response.get("id") != request_id:
            raise RuntimeError("MediaPipe sidecar response was out of order")
        return [
            [_Landmark(float(x), float(y)) for x, y in hand]
            for hand in response.get("hands", [])
        ]

    def close(self) -> None:
        process = self._process
        if process.poll() is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


class HandGestureDetector:
    """Detect a deliberate raised/open palm without blocking video inference."""

    def __init__(
        self,
        *,
        interval_s: float = 0.10,
        max_candidates: int = 4,
        max_crop_side: int = 448,
        emit_cooldown_s: float = 20.0,
        confirm_hits: int = 2,
        release_s: float = 1.0,
        min_confidence: float = 0.70,
        max_hand_center_y: float = 0.68,
        min_palm_width_ratio: float = 0.20,
    ) -> None:
        self.interval_s = max(0.10, interval_s)
        self.max_candidates = max(1, max_candidates)
        self.max_crop_side = max(160, max_crop_side)
        self.emit_cooldown_s = max(3.0, emit_cooldown_s)
        self.confirm_hits = max(1, confirm_hits)
        self.release_s = max(0.4, release_s)
        self.min_confidence = min(0.95, max(0.50, float(min_confidence)))
        self.max_hand_center_y = min(0.90, max(0.35, max_hand_center_y))
        self.min_palm_width_ratio = min(
            0.60, max(0.10, min_palm_width_ratio)
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hamy-hand-gesture"
        )
        self._future: Future | None = None
        self._last_submit_s = float("-inf")
        self._last_emit_s: dict[tuple[str, int], float] = {}
        self._hits: dict[tuple[str, int], tuple[int, float]] = {}
        self._latched: set[tuple[str, int]] = set()
        self._missing_since_s: dict[tuple[str, int], float] = {}
        self._model = None
        self._model_lock = threading.Lock()
        self._candidate_cursor = 0
        self._backend, self._sidecar_command = self._probe_mediapipe()
        self._available = self._backend is not None
        # Start MediaPipe during application boot, not on the user's first
        # wave. This removes the one-time model startup pause from interaction.
        if self._available:
            try:
                self._ensure_model()
            except Exception as error:  # Gesture remains an optional subsystem.
                print(f"[Bé Xinh/Gesture] startup failed: {error}")
                self._available = False

    @classmethod
    def from_env(cls) -> "HandGestureDetector":
        return cls(
            interval_s=_env_float("HAMY_GESTURE_INTERVAL_SECONDS", 0.10, 0.10),
            max_candidates=_env_int("HAMY_GESTURE_MAX_CANDIDATES", 4, 1),
            max_crop_side=_env_int("HAMY_GESTURE_CROP_SIDE", 448, 192),
            emit_cooldown_s=_env_float(
                "HAMY_GESTURE_COOLDOWN_SECONDS", 20.0, 3.0
            ),
            confirm_hits=_env_int("HAMY_GESTURE_CONFIRM_HITS", 2, 1),
            release_s=_env_float("HAMY_GESTURE_RELEASE_SECONDS", 1.0, 0.4),
            min_confidence=_env_float(
                "HAMY_GESTURE_MIN_CONFIDENCE", 0.70, 0.50
            ),
            max_hand_center_y=_env_float(
                "HAMY_GESTURE_MAX_HAND_CENTER_Y", 0.68, 0.35
            ),
            min_palm_width_ratio=_env_float(
                "HAMY_GESTURE_MIN_PALM_WIDTH_RATIO", 0.20, 0.10
            ),
        )

    @property
    def available(self) -> bool:
        return self._available

    def submit(
        self,
        channel: str,
        frame_bgr: np.ndarray | None,
        tracks: Iterable[Track],
        now_s: float,
    ) -> bool:
        if not self._available or frame_bgr is None:
            return False
        if self._future is not None and not self._future.done():
            return False
        if now_s - self._last_submit_s < self.interval_s:
            return False

        candidates = self._make_candidates(channel, frame_bgr, tracks)
        if not candidates:
            return False

        self._last_submit_s = now_s
        self._future = self._executor.submit(self._detect, candidates, now_s)
        return True

    def poll(self) -> list[GestureEvent]:
        future = self._future
        if future is None or not future.done():
            return []
        self._future = None
        try:
            raw_events: list[_RawPalm] = future.result()
        except Exception as error:  # noqa: BLE001 - optional feature only
            print(f"[Bé Xinh/Gesture] skipped: {error}")
            return []

        emitted: list[GestureEvent] = []
        seen_keys = {(raw.channel, raw.global_id) for raw in raw_events}
        latest_event_s = max((raw.now_s for raw in raw_events), default=self._last_submit_s)

        # A held-up hand is one gesture, not a new greeting every cooldown.
        # Re-arm only after the palm has disappeared continuously.
        for key in list(self._latched):
            if key in seen_keys:
                self._missing_since_s.pop(key, None)
                continue
            missing_since = self._missing_since_s.setdefault(key, latest_event_s)
            if latest_event_s - missing_since >= self.release_s:
                self._latched.discard(key)
                self._missing_since_s.pop(key, None)
                self._hits.pop(key, None)

        for raw in raw_events:
            key = (raw.channel, raw.global_id)
            self._missing_since_s.pop(key, None)
            if key in self._latched:
                continue
            count, previous_s = self._hits.get(key, (0, float("-inf")))
            if raw.now_s - previous_s <= 1.2:
                count += 1
            else:
                count = 1
            self._hits[key] = (count, raw.now_s)

            last_emit = self._last_emit_s.get(key, float("-inf"))
            if count < self.confirm_hits or raw.now_s - last_emit < self.emit_cooldown_s:
                continue

            self._last_emit_s[key] = raw.now_s
            self._hits[key] = (0, raw.now_s)
            self._latched.add(key)
            emitted.append(
                GestureEvent(
                    kind="OPEN_PALM",
                    channel=raw.channel,
                    global_id=raw.global_id,
                    now_s=raw.now_s,
                    confidence=raw.confidence,
                )
            )

        # Expire half-completed gestures so a random hand pose minutes later
        # cannot complete an old hit.
        for key, (_count, hit_s) in list(self._hits.items()):
            if key not in seen_keys and latest_event_s - hit_s > 2.0:
                self._hits.pop(key, None)
        return emitted

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        model = self._model
        self._model = None
        if model is not None:
            try:
                model.close()
            except Exception:
                pass

    @staticmethod
    def _probe_mediapipe() -> tuple[str | None, list[str] | None]:
        try:
            import mediapipe as mp

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                print("[Bé Xinh/Gesture] open-palm detector ready (async).")
                return "inprocess", None
        except Exception:
            pass

        configured = os.getenv("HAMY_GESTURE_PYTHON", "").strip()
        commands: list[list[str]] = []
        if configured:
            commands.append([configured])
        py_launcher = shutil.which("py")
        if py_launcher:
            commands.append([py_launcher, "-3.12"])
        python312 = shutil.which("python3.12")
        if python312:
            commands.append([python312])
        probe = (
            "import mediapipe as mp; "
            "assert hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands')"
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        for command in commands:
            try:
                checked = subprocess.run(
                    [*command, "-c", probe],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15.0,
                    check=False,
                    creationflags=creationflags,
                )
                if checked.returncode == 0:
                    print(
                        "[Bé Xinh/Gesture] open-palm detector ready "
                        "(Python 3.12 sidecar, async)."
                    )
                    return "sidecar", command
            except (OSError, subprocess.TimeoutExpired):
                continue

        print(
            "[Bé Xinh/Gesture] MediaPipe unavailable -> gesture disabled. "
            "Install mediapipe on Python 3.12."
        )
        return None, None

    @staticmethod
    def _probe_mediapipe_legacy() -> bool:
        try:
            import mediapipe  # noqa: F401
        except Exception:
            print(
                "[Bé Xinh/Gesture] MediaPipe chưa có -> gesture tạm tắt. "
                "Cài: pip install mediapipe"
            )
            return False
        print("[Bé Xinh/Gesture] open-palm detector ready (async).")
        return True

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            if self._backend == "sidecar":
                if not self._sidecar_command:
                    raise RuntimeError("MediaPipe sidecar command is missing")
                self._model = _MediaPipeSidecar(self._sidecar_command)
            else:
                import mediapipe as mp

                self._model = mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    model_complexity=0,
                    min_detection_confidence=0.50,
                    min_tracking_confidence=0.50,
                )
            return self._model

    def _make_candidates(
        self,
        channel: str,
        frame: np.ndarray,
        tracks: Iterable[Track],
    ) -> list[_Candidate]:
        h, w = frame.shape[:2]
        ranked = sorted(tracks, key=lambda track: track.bbox.area, reverse=True)
        # Rotate through the largest few people so a person waving from farther
        # away is not permanently starved by somebody seated close to camera.
        pool = ranked[: max(self.max_candidates, min(6, len(ranked)))]
        if pool:
            start = self._candidate_cursor % len(pool)
            ordered = pool[start:] + pool[:start]
            self._candidate_cursor = (start + self.max_candidates) % len(pool)
        else:
            ordered = []
        candidates: list[_Candidate] = []
        for track in ordered[: self.max_candidates]:
            bbox = track.bbox
            # MediaPipe works much better when the hand occupies a useful
            # fraction of the crop. For greeting gestures we only need the
            # upper body, and extra side/top padding keeps a raised hand that
            # extends beyond the YOLO torso box.
            pad_x = int(max(16.0, bbox.width * 0.35))
            pad_top = int(max(16.0, bbox.height * 0.30))
            x1 = max(0, int(bbox.x1) - pad_x)
            y1 = max(0, int(bbox.y1) - pad_top)
            x2 = min(w, int(bbox.x2) + pad_x)
            y2 = min(h, int(bbox.y1 + bbox.height * 0.90))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            longest = max(ch, cw)
            if longest > self.max_crop_side:
                scale = self.max_crop_side / longest
                crop = cv2.resize(
                    crop,
                    (max(1, round(cw * scale)), max(1, round(ch * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            gid = int(track.global_person_id or track.track_id)
            candidates.append(_Candidate(channel, gid, crop.copy()))
        return candidates

    def _detect(self, candidates: list[_Candidate], now_s: float) -> list[_RawPalm]:
        model = self._ensure_model()
        found: list[_RawPalm] = []
        for candidate in candidates:
            if self._backend == "sidecar":
                detected_hands = model.process(candidate.crop)
            else:
                rgb = cv2.cvtColor(candidate.crop, cv2.COLOR_BGR2RGB)
                result = model.process(rgb)
                detected_hands = [
                    hand.landmark for hand in (result.multi_hand_landmarks or [])
                ]
            for landmarks in detected_hands:
                confidence = self._open_palm_confidence(landmarks)
                if (
                    confidence >= self.min_confidence
                    and self._is_raised_front_palm(landmarks)
                ):
                    found.append(
                        _RawPalm(
                            channel=candidate.channel,
                            global_id=candidate.global_id,
                            confidence=confidence,
                            now_s=now_s,
                        )
                    )
                    break
        return found

    def _is_raised_front_palm(self, landmarks) -> bool:
        """Reject desk-level and edge-on hands that look open by accident."""
        if len(landmarks) < 21:
            return False

        def dist(a: int, b: int) -> float:
            dx = float(landmarks[a].x) - float(landmarks[b].x)
            dy = float(landmarks[a].y) - float(landmarks[b].y)
            return math.hypot(dx, dy)

        palm_center_y = sum(float(landmarks[index].y) for index in (0, 5, 9, 13, 17)) / 5.0
        hand_length = dist(0, 12)
        palm_width = dist(5, 17)
        return (
            palm_center_y <= self.max_hand_center_y
            and hand_length > 1e-6
            and palm_width / hand_length >= self.min_palm_width_ratio
        )

    @staticmethod
    def _open_palm_confidence(landmarks) -> float:
        """Orientation-tolerant open-hand heuristic from 21 hand landmarks."""
        if len(landmarks) < 21:
            return 0.0

        def dist(a: int, b: int) -> float:
            dx = landmarks[a].x - landmarks[b].x
            dy = landmarks[a].y - landmarks[b].y
            return math.hypot(dx, dy)

        wrist = 0
        # Thumb geometry changes a lot with hand rotation. Treat the four long
        # fingers as the strongest signal and use the thumb as supporting
        # evidence, so a real five-finger wave is not rejected by perspective.
        long_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
        long_extended = sum(
            1 for tip, joint in long_pairs
            if dist(wrist, tip) >= dist(wrist, joint) * 1.06
        )
        thumb_extended = (
            1.0 if dist(wrist, 4) >= dist(wrist, 3) * 1.04 else 0.0
        )

        # Open palms are also spread. The hand-center term is deliberately only
        # a small bonus: a user can wave from chest/shoulder height and should
        # still be recognised.
        spread = dist(8, 20)
        spread_factor = min(1.0, spread / 0.24)
        hand_center_y = sum(point.y for point in landmarks) / len(landmarks)
        raised_bonus = 1.0 if hand_center_y <= 0.72 else 0.0
        return (
            (long_extended / 4.0) * 0.68
            + thumb_extended * 0.12
            + spread_factor * 0.15
            + raised_bonus * 0.05
        )


__all__ = ["GestureEvent", "HandGestureDetector"]
