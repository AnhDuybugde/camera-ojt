"""One employee owns one global identity across overlapping cameras."""
import numpy as np

from camera_tracking.domain import BoundingBox, Track
from camera_tracking.tracking.global_identity import (
    GlobalIdentityConfig,
    GlobalIdentityManager,
)


class ColorEmb:
    def extract(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        value = crop_bgr.mean(axis=(0, 1)).astype(np.float32)
        norm = float(np.linalg.norm(value))
        return value / norm if norm else None


def _raw(tid: int, box: BoundingBox) -> Track:
    return Track(tid, box, 0.9, age=1, hits=2, confirmed=True)


def _frame() -> np.ndarray:
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[40:160, 20:120] = (0, 0, 255)  # do
    frame[40:160, 260:360] = (0, 0, 255)  # do (giong het de ReID muon gop)
    return frame


def _manager() -> GlobalIdentityManager:
    return GlobalIdentityManager(ColorEmb(), GlobalIdentityConfig())


def test_same_person_in_camera_overlap_keeps_one_global_id() -> None:
    manager = _manager()
    frame = _frame()
    left = BoundingBox(20, 40, 120, 160)
    right = BoundingBox(260, 40, 360, 160)
    first = manager.update(
        channel="A", frame=frame, tracks=[_raw(10, left)], now_s=0.0)
    assert first[0].track_id == 1
    manager.bind_employee(1, "E1")

    second = manager.update(
        channel="B", frame=frame, tracks=[_raw(20, right)], now_s=0.5)
    assert second[0].track_id == 1
    assert second[0].employee_id == "E1"
    assert set(manager.identities[1].sightings) == {"A", "B"}


def test_batch_does_not_mark_first_camera_lost_at_same_timestamp() -> None:
    manager = _manager()
    frame = _frame()
    result = manager.update_batch({
        "A": (frame, [_raw(10, BoundingBox(20, 40, 120, 160))]),
        "B": (frame, []),
    }, now_s=1.0)
    assert result["A"][0].track_id == 1
    assert manager.state_of(1).value == "ACTIVE"


def test_named_id_reusable_after_window() -> None:
    manager = _manager()
    frame = _frame()
    left = BoundingBox(20, 40, 120, 160)
    manager.update(channel="A", frame=frame,
                   tracks=[_raw(10, left)], now_s=0.0)
    manager.bind_employee(1, "E1")
    # 10s sau (qua window 3s): nguoi that di A->B duoc noi ve G1.
    moved = manager.update(channel="B", frame=frame,
                           tracks=[_raw(21, left)], now_s=10.0)
    assert moved[0].track_id == 1
    assert moved[0].employee_id == "E1"


def test_unbind_employee() -> None:
    manager = _manager()
    frame = _frame()
    manager.update(channel="A", frame=frame,
                   tracks=[_raw(10, BoundingBox(20, 40, 120, 160))],
                   now_s=0.0)
    manager.bind_employee(1, "E1")
    assert manager.unbind_employee(1) is True
    assert manager.employee_id_of(1) is None
    assert manager.unbind_employee(1) is False
    assert manager.unbind_employee(999) is False


def test_binding_employee_cannot_steal_existing_owner() -> None:
    manager = _manager()
    frame = _frame()
    manager.update(channel="A", frame=frame,
                   tracks=[_raw(10, BoundingBox(20, 40, 120, 160))],
                   now_s=0.0)
    manager.bind_employee(1, "E1")
    other = np.zeros_like(frame)
    other[40:160, 260:360] = (255, 0, 0)
    second = manager.update(channel="B", frame=other,
                            tracks=[_raw(20, BoundingBox(260, 40, 360, 160))],
                            now_s=0.5)
    assert second[0].track_id == 2
    assert manager.bind_employee(2, "E1") is False
    assert manager.employee_id_of(1) == "E1"
    assert manager.employee_id_of(2) is None
