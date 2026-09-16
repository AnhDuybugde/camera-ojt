"""Face async worker: tách Face inference khỏi main realtime loop.

Trước đây main loop gọi InsightFace đồng bộ + giữ ``face_model_lock``
200-500ms nên toàn pipeline lag theo:

    YOLO -> Track -> FACE (chờ 300ms) -> YOLO frame tiếp theo

Kiến trúc mới (event-driven):

    Main loop (realtime, không bao giờ chờ Face)
        YOLO batch A+B -> ByteTrack -> GlobalID fast-path -> workstate
          └── cần face? (track mới / chưa biết tên / hết recheck)
                └── face_job_queue.put((channel, gid, crop_nhỏ, track_copy))

    FaceWorker (background thread, GPU)
        queue -> InsightFace detect_embed(crop) -> matcher -> consensus
          └── result_queue -> main loop poll + bind/attendance/greet

Nguyên tắc realtime: queue huu han, khong block main loop. Moi
``(channel, gid)`` chi co mot job dang cho/xu ly; khi day thi giu cac job da
nhan thay vi xoa job cu, tranh mot nhom track bi doi vo han.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from camera_tracking.domain import BoundingBox, Track


@dataclass(slots=True)
class FaceJob:
    channel: str
    gid: int
    crop_bgr: np.ndarray
    track: Track
    day_str: str
    wall_iso: str
    time_tag: str
    now_s: float
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class FaceResult:
    channel: str
    gid: int
    track: Track
    crop_bgr: np.ndarray
    detection: object
    match: object
    sharpness: float
    quality: float
    identity_confirmed: bool
    day_str: str
    wall_iso: str
    time_tag: str
    now_s: float


class FaceWorker:
    """Background Face inference, không block main loop."""

    def __init__(
        self,
        consumer,
        *,
        max_queue: int = 8,
        model_lock: threading.Lock | None = None,
        max_job_age_s: float = 2.0,
    ) -> None:
        self.consumer = consumer
        self.max_queue = max(1, int(max_queue))
        self.model_lock = model_lock or threading.Lock()
        self.max_job_age_s = max(0.5, float(max_job_age_s))
        self._jobs: queue.Queue[FaceJob] = queue.Queue(maxsize=self.max_queue)
        self._results: queue.Queue[FaceResult] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Once-per-track: gid đã có kết quả known thì không enqueue lại
        # cho tới khi hết recheck hoặc mất track lâu.
        self._known_at: dict[tuple[str, int], float] = {}
        self._last_attempt: dict[tuple[str, int], float] = {}
        self._pending: set[tuple[str, int]] = set()
        self._lock = threading.Lock()

    # -- lifecycle --
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._drain, name="face-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # -- gating (gọi từ main loop, rẻ, không inference) --
    def need_face(
        self,
        gid: int,
        *,
        known: bool,
        now_s: float,
        recheck_s: float = 30.0,
        unknown_cooldown_s: float = 1.0,
        channel: str = "*",
    ) -> bool:
        """Track này có cần enqueue Face không?

        - Đã biết tên -> chỉ recheck sau ``recheck_s`` (mặc định 30s).
        - Chưa biết -> cooldown ngắn để thử lại khi góc mặt đẹp hơn.
        """
        with self._lock:
            key = (channel, gid)
            if known:
                last = self._known_at.get(key, float("-inf"))
                return (now_s - last) >= max(0.0, recheck_s)
            last_try = self._last_attempt.get(key, float("-inf"))
            return (now_s - last_try) >= max(0.0, unknown_cooldown_s)

    def mark_attempt(self, gid: int, now_s: float, channel: str = "*") -> None:
        with self._lock:
            self._last_attempt[(channel, gid)] = now_s

    def mark_known(self, gid: int, now_s: float, channel: str = "*") -> None:
        with self._lock:
            self._known_at[(channel, gid)] = now_s

    def forget_retired(self, alive: set[int]) -> None:
        with self._lock:
            for key in [k for k in self._known_at if k[1] not in alive]:
                self._known_at.pop(key, None)
            for key in [k for k in self._last_attempt if k[1] not in alive]:
                self._last_attempt.pop(key, None)
            for key in [k for k in self._pending if k[1] not in alive]:
                self._pending.discard(key)

    def submit(self, job: FaceJob) -> bool:
        """Enqueue khong block; khong xoa job cu va khong lap cung track."""
        key = (job.channel, job.gid)
        with self._lock:
            if key in self._pending:
                return False
            self._pending.add(key)
        try:
            self._jobs.put_nowait(job)
            return True
        except queue.Full:
            with self._lock:
                self._pending.discard(key)
            return False

    def poll_results(self) -> list[FaceResult]:
        out: list[FaceResult] = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    # -- worker --
    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                # Drop job cũ: mặt trong crop 2s trước không còn giá trị.
                if time.monotonic() - job.enqueued_at > self.max_job_age_s:
                    continue
                observations = self._run_job(job)
                for obs in observations:
                    try:
                        self._results.put_nowait(obs)
                    except queue.Full:
                        break
            except Exception:
                continue
            finally:
                with self._lock:
                    self._pending.discard((job.channel, job.gid))
                try:
                    self._jobs.task_done()
                except ValueError:
                    pass

    def _run_job(self, job: FaceJob) -> list[FaceResult]:
        from camera_tracking.domain import Frame, TrackEvent
        from camera_tracking.face.consumer import _crop as _crop_fn  # noqa

        # Dựng TrackEvent giả với frame=crop để tái dùng FaceTrackConsumer
        # (gating + quality + consensus đã có sẵn, đã được test).
        # Bbox chuyển về toạ độ crop (origin 0,0).
        h, w = job.crop_bgr.shape[:2]
        x1 = max(0, min(w, round(job.track.bbox.x1)))
        y1 = max(0, min(h, round(job.track.bbox.y1)))
        # crop_bgr đã là person crop nên track box trong crop ~ full crop.
        # Dùng bbox full-crop để consumer không loại do area.
        crop_track = Track(
            track_id=job.gid,
            bbox=BoundingBox(0, 0, float(job.crop_bgr.shape[1]),
                             float(job.crop_bgr.shape[0])),
            confidence=job.track.confidence,
            age=job.track.age,
            hits=job.track.hits,
            confirmed=job.track.confirmed,
            local_track_id=job.track.local_track_id,
            global_person_id=job.gid,
            employee_id=job.track.employee_id,
        )
        event = TrackEvent(
            channel=job.channel,
            frame=Frame(0, job.now_s, job.crop_bgr),
            tracks=(crop_track,),
        )
        with self.model_lock:
            observations = self.consumer.consume(event, job.now_s)
        results: list[FaceResult] = []
        for obs in observations:
            # Trả track gốc (toạ độ full-frame) để main loop vẽ/bind đúng.
            results.append(FaceResult(
                channel=job.channel,
                gid=job.gid,
                track=job.track,
                crop_bgr=job.crop_bgr,
                detection=obs.detection,
                match=obs.match,
                sharpness=obs.sharpness,
                quality=obs.quality,
                identity_confirmed=obs.identity_confirmed,
                day_str=job.day_str,
                wall_iso=job.wall_iso,
                time_tag=job.time_tag,
                now_s=job.now_s,
            ))
        # Giữ face_marks tương thích: detection bbox đang tương đối theo crop,
        # main loop sẽ cộng offset person (x1,y1) khi vẽ.
        _ = (x1, y1)
        return results


__all__ = ["FaceJob", "FaceResult", "FaceWorker"]
