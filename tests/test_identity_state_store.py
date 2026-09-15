import numpy as np

from camera_tracking.domain import BoundingBox, Track
from camera_tracking.tracking.global_identity import (
    GlobalIdentityConfig,
    GlobalIdentityManager,
)
from camera_tracking.tracking.state_store import DailyIdentityStore


class _Embedding:
    def extract(self, crop):
        return np.array([1.0, 0.0], dtype=np.float32)


def _track(track_id: int = 1) -> Track:
    return Track(track_id, BoundingBox(0, 0, 20, 40), 0.9, 1, 1, True)


def test_tentative_tracks_do_not_consume_public_ids() -> None:
    manager = GlobalIdentityManager(
        _Embedding(), GlobalIdentityConfig(tentative_min_hits=3)
    )
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    assert manager.update(channel="A", frame=frame, tracks=[_track()], now_s=0) == []
    assert manager.update(channel="A", frame=frame, tracks=[_track()], now_s=1) == []
    promoted = manager.update(channel="A", frame=frame, tracks=[_track()], now_s=2)
    assert promoted[0].track_id == 1
    assert set(manager.identities) == {1}


def test_daily_identity_store_restores_gid_and_binding(tmp_path) -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    manager = GlobalIdentityManager(_Embedding())
    manager.update(channel="A", frame=frame, tracks=[_track()], now_s=0)
    assert manager.bind_employee(1, "E1") is True
    store = DailyIdentityStore(tmp_path / "identity.db", "osnet:test")
    store.save("2026-09-14", manager)

    restored = GlobalIdentityManager(_Embedding())
    assert store.load("2026-09-14", restored) == 1
    assert restored.employee_id_of(1) == "E1"
    next_track = restored.update(
        channel="A", frame=frame, tracks=[_track(2)], now_s=1
    )
    assert next_track[0].track_id == 1
