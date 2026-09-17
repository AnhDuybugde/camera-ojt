"""FaceWorker: async once-per-track, không block main loop."""
import threading
import time

import numpy as np

from camera_tracking.domain import BoundingBox, Track
from camera_tracking.face.worker import FaceJob, FaceWorker


class _EmptyConsumer:
    def consume(self, event, now_s):
        return []


def _job(gid: int = 1, now_s: float = 0.0,
         priority=(2.0, 0.0, 0.0)) -> FaceJob:
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    track = Track(gid, BoundingBox(0, 0, 64, 64), 0.9, 3, 3, True,
                  local_track_id=gid, global_person_id=gid)
    return FaceJob(channel="A", gid=gid, crop_bgr=crop, track=track,
                   day_str="2026-01-01", wall_iso="2026-01-01T00:00:00",
                   time_tag="000000", now_s=now_s, priority=priority)


def test_need_face_once_per_track_with_recheck() -> None:
    worker = FaceWorker(_EmptyConsumer(), max_queue=2)
    assert worker.need_face(1, known=False, now_s=0.0) is True
    worker.mark_attempt(1, 0.0)
    assert worker.need_face(1, known=False, now_s=0.5,
                            unknown_cooldown_s=1.0) is False
    assert worker.need_face(1, known=False, now_s=1.5,
                            unknown_cooldown_s=1.0) is True
    worker.mark_known(1, 10.0)
    assert worker.need_face(1, known=True, now_s=15.0, recheck_s=30.0) is False
    assert worker.need_face(1, known=True, now_s=45.0, recheck_s=30.0) is True


def test_submit_keeps_accepted_jobs_when_full() -> None:
    worker = FaceWorker(_EmptyConsumer(), max_queue=2)
    assert worker.submit(_job(1)) is True
    assert worker.submit(_job(2)) is True
    # Queue day -> tu choi job moi, khong lam mat hai job da chap nhan.
    assert worker.submit(_job(3)) is False
    assert worker._jobs.qsize() == 2


def test_submit_coalesces_same_camera_and_gid() -> None:
    worker = FaceWorker(_EmptyConsumer(), max_queue=2)
    assert worker.submit(_job(1)) is True
    assert worker.submit(_job(1)) is False
    assert worker._jobs.qsize() == 1


def test_foreground_job_is_drained_before_background() -> None:
    worker = FaceWorker(_EmptyConsumer(), max_queue=3)
    assert worker.submit(_job(2, priority=(1.0, 0.0)))
    assert worker.submit(_job(1, priority=(0.0, 0.0)))
    assert worker._jobs.get_nowait().job.gid == 1


def test_cooldown_is_independent_per_camera() -> None:
    worker = FaceWorker(_EmptyConsumer(), max_queue=2)
    worker.mark_attempt(1, 0.0, "A")
    assert worker.need_face(1, known=False, now_s=0.1,
                            unknown_cooldown_s=1.0, channel="A") is False
    assert worker.need_face(1, known=False, now_s=0.1,
                            unknown_cooldown_s=1.0, channel="B") is True


def test_worker_drains_without_blocking_main() -> None:
    worker = FaceWorker(_EmptyConsumer(), max_queue=2)
    worker.start()
    try:
        assert worker.submit(_job(9)) is True
        deadline = time.monotonic() + 2.0
        while worker._jobs.qsize() > 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert worker._jobs.qsize() == 0
        assert worker.poll_results() == []
    finally:
        worker.stop()


def test_worker_reports_job_failure_and_keeps_running() -> None:
    class _BrokenConsumer:
        def consume(self, event, now_s):
            raise ValueError("bad face frame")

    worker = FaceWorker(_BrokenConsumer(), max_queue=2)
    worker.start()
    try:
        assert worker.submit(_job(10)) is True
        deadline = time.monotonic() + 2.0
        while worker.jobs_failed == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert worker.jobs_failed == 1
        assert worker.last_error == "ValueError: bad face frame"
        assert worker.ensure_alive() is True
    finally:
        worker.stop()


def test_forget_retired() -> None:
    worker = FaceWorker(_EmptyConsumer())
    worker.mark_attempt(7, 0.0)
    worker.mark_known(7, 0.0)
    worker.forget_retired(set())
    assert worker.need_face(7, known=False, now_s=0.0) is True


def test_model_lock_shared() -> None:
    lock = threading.Lock()
    worker = FaceWorker(_EmptyConsumer(), model_lock=lock)
    assert worker.model_lock is lock
