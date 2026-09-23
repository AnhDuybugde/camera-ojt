"""Critical boundaries of the unified flow; no cameras, model downloads or cloud."""
from datetime import datetime
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps/attendance"))

from camera_tracking.api.service import ApplicationAPI
from camera_tracking.api.supabase_auth import SupabaseAuth
from camera_tracking.store.embeddings import EmbeddingStore
from camera_tracking.store.event_log import EventSync
from camera_tracking.store.replication import SQLiteReplica
from camera_tracking.workstate.door import DoorTransitions
from attendance.attendance_service import AttendanceService
from database.db import Database


@pytest.fixture
def business(tmp_path):
    db = Database(tmp_path / "platform.db")
    for employee in ("NV01", "NV02"):
        db.add_employee({"employee_id": employee, "full_name": employee})
        db.save_work_schedule(employee, "2026-09-21", "ON", "MORNING")
    return db


def test_directional_attendance_atomic_duplicate_and_offline_retry(business):
    service = AttendanceService(business)
    event = str(uuid4())
    observed = datetime.fromisoformat("2026-09-21T08:00:00+07:00")
    assert service.record_direction("NV01", observed, direction="CHECK_IN", event_id=event).action == "CHECK_IN"
    assert service.record_direction("NV01", observed, direction="CHECK_IN", event_id=event).action == "DUPLICATE"
    assert business.fetch_one("SELECT COUNT(*) AS n FROM event_outbox")["n"] == 1
    calls = []
    def unavailable(row):
        raise ConnectionError()
    assert EventSync(business, unavailable, clock=lambda: 10).flush() == 0
    assert business.fetch_one("SELECT attempts FROM event_outbox")["attempts"] == 1
    assert EventSync(business, calls.append, clock=lambda: 100).flush() == 1
    assert len(calls) == 1 and calls[0]["event_id"] == event


def test_door_requires_observed_direction_and_does_not_infer_exit():
    door = DoorTransitions(dwell_s=0.1, max_gap_s=1)
    def stable(zone, at):
        door.update(1, zone, at)
        return door.update(1, zone, at + 0.2)
    assert stable("inside", 0) is None
    assert stable("door", 0.3) is None
    assert stable("outside", 0.6) == "CHECK_OUT"
    assert stable("outside", 0.9) is None
    door.expire(3)
    assert stable("inside", 4) is None


def test_api_enforces_owner_and_does_not_accept_sql(business, monkeypatch):
    class Auth:
        def login(self, email, password):
            assert email == "nv01@example.com" and password == "employee-password"
            return {"role": "EMPLOYEE", "employee_id": "NV01", "user_id": "user-01",
                    "email": email, "expires_in": 3600}
    api = ApplicationAPI(business, Auth(), None, None)
    token = api.login("nv01@example.com", "employee-password")["token"]
    assert len(api.dispatch("employees", "list_employees", [], {}, token)) == 1
    with pytest.raises(PermissionError):
        api.dispatch("employees", "get_employee", ["NV02"], {}, token)
    with pytest.raises(PermissionError):
        api.dispatch("employees", "execute", ["DELETE FROM employees"], {}, token)
    with pytest.raises(PermissionError):
        api.dispatch("employees", "save_work_schedules", [[("NV02", "2026-09-21", "MORNING", "OFF")]],
                     {"actor_role": "SYSTEM"}, token)


def test_supabase_auth_maps_role_without_returning_tokens():
    import io
    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *_): self.close()
    def opener(request, timeout):
        if "/token?" in request.full_url:
            return Response(b'{"access_token":"not-returned","expires_in":3600,'
                            b'"user":{"id":"u1","email":"e@example.com"}}')
        return Response(b'[{"user_id":"u1","employee_id":"NV01","role":"employee","active":true}]')
    auth = SupabaseAuth("https://example.supabase.co", "anon", "service", opener)
    identity = auth.login("e@example.com", "pw")
    assert identity["role"] == "EMPLOYEE" and identity["employee_id"] == "NV01"
    assert "access_token" not in identity


def test_versioned_gallery_keeps_all_samples_and_replaces_atomically(tmp_path):
    store = EmbeddingStore(tmp_path / "enrollment.db")
    samples = [np.eye(32, dtype=np.float32)[index] for index in range(29)]
    store.replace("NV01", "model-a", samples)
    assert len(store.samples("model-a")) == 29
    assert store.samples("model-b") == []
    with pytest.raises(ValueError):
        store.replace("NV01", "model-a", [samples[0], np.zeros(32)])
    assert len(store.samples("model-a")) == 29


def test_cloud_conflict_preserves_local_edit(business):
    replica = SQLiteReplica(business.path)
    replica.bootstrap()
    class Remote:
        def compare_batch(self, rows):
            return [{"conflict": True, "record": {
                "kind": row["kind"], "record_key": row["record_key"],
                "payload": {}, "revision": 8, "deleted": False, "change_seq": 1,
            }} for row in rows]
        def changes(self, cursor):
            return []
    replica.sync(Remote())
    assert business.get_employee("NV01")["full_name"] == "NV01"
    assert business.fetch_one("SELECT COUNT(*) AS n FROM record_conflicts")["n"] > 0
    assert business.fetch_one("SELECT COUNT(*) AS n FROM record_outbox")["n"] > 0
