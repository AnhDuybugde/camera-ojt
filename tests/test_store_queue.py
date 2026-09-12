"""Test WriteQueue (SQLite) + FaceCropSaver best-shot (can cv2)."""
import numpy as np
import pytest

from camera_tracking.store.queue import WriteQueue

cv2 = pytest.importorskip("cv2")


def test_queue_push_peek_ack(tmp_path) -> None:
    queue = WriteQueue(tmp_path / "q.db")
    assert len(queue) == 0
    queue.push("attendance", {"date": "2026-09-12", "person_id": "An"})
    queue.push("event", {"date": "2026-09-12", "event": "CHECK_IN"})
    assert len(queue) == 2
    rows = queue.peek(10)
    assert rows[0][1] == "attendance" and rows[0][2]["person_id"] == "An"
    queue.ack(rows[0][0])
    assert len(queue) == 1
    queue.bump(rows[1][0])


def test_crop_saver_keeps_best_only(tmp_path) -> None:
    from camera_tracking.store.faces import FaceCropSaver

    saver = FaceCropSaver(root=tmp_path / "faces", max_per_owner_day=5)
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    first = saver.save_best_crop(day="2026-09-12", owner="An", known=True,
                                 global_id=1, image_bgr=img, face_score=0.8,
                                 sharpness=200.0, time_tag="080000")
    assert first is not None
    # Diem thap hon -> khong ghi de (chong spam file).
    second = saver.save_best_crop(day="2026-09-12", owner="An", known=True,
                                  global_id=1, image_bgr=img, face_score=0.5,
                                  sharpness=50.0, time_tag="080100")
    assert second is None
