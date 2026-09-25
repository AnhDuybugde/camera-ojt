"""Connection flow: UI contract -> backend :8767 -> pipeline :8765.

Hermetic: temp DB, ephemeral ports, stub face detector. No camera, GPU,
Supabase network or Streamlit runtime needed.
"""
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from camera_tracking.api.codec import decode, encode
from camera_tracking.api.server import make_server
from camera_tracking.api.service import ApplicationAPI


class _StubDetector:
    def detect(self, frame):
        return []


def _post(base, service, method, args=None, kwargs=None, token=""):
    request = urllib.request.Request(
        f"{base}/api/v1/{service}/{method}",
        data=json.dumps(encode({"args": args or [], "kwargs": kwargs or {}})).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, decode(json.load(response)["result"])
    except Exception as error:
        from urllib.error import HTTPError
        if isinstance(error, HTTPError):
            return error.code, json.load(error)
        raise


@pytest.fixture
def live_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMERA_DATABASE_PATH", str(tmp_path / "conn.db"))
    monkeypatch.setenv("CAMERA_ADMIN_PASSWORD", "conn-check-password")
    # Keep Supabase out of this hermetic check: local auth only.
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps/attendance"))
    try:
        from camera_tracking.application.backend import load_services
        from camera_tracking.api.operations import Operations
        from camera_tracking.api.enrollment import Enrollment
        db, auth, attendance, admin, sync = load_services()
        api = ApplicationAPI(db, auth, admin, sync,
                             Operations(db, attendance, db.path),
                             Enrollment(_StubDetector()))
    finally:
        sys.path.remove(str(Path(__file__).resolve().parents[1] / "apps/attendance"))
    server = make_server(api, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", api
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def test_backend_listens_and_rejects_anonymous(live_backend):
    base, _ = live_backend
    code, _ = _post(base, "operations", "diagnostics")
    assert code == 403  # listening, but token required


def test_login_then_dispatch_employees_and_diagnostics(live_backend):
    base, _ = live_backend
    code, session = _post(base, "auth", "login", ["ADMIN", "conn-check-password"])
    assert code == 200 and session["token"]
    code, employees = _post(base, "employees", "list_employees", token=session["token"])
    assert code == 200 and isinstance(employees, list)
    code, health = _post(base, "operations", "diagnostics", token=session["token"])
    assert code == 200
    assert {"pending_events", "pending_records", "conflicts"} <= set(health)


def test_api_never_exposes_sql_or_cross_employee_reads(live_backend):
    base, _ = live_backend
    code, session = _post(base, "auth", "login", ["ADMIN", "conn-check-password"])
    token = session["token"]
    code, _ = _post(base, "employees", "execute", ["DELETE FROM employees"], token=token)
    assert code in (400, 403)
    code, _ = _post(base, "employees", "get_employee", ["no-such-id"], token=token)
    assert code in (200, 400)  # device-owned errors stay 4xx, never 500


def test_enrollment_endpoint_wired_without_models(live_backend):
    base, _ = live_backend
    code, session = _post(base, "auth", "login", ["ADMIN", "conn-check-password"])
    code, _ = _post(base, "enrollment", "detect", ["not-an-image"], token=session["token"])
    assert code == 400  # invalid image rejected cleanly, endpoint exists


def test_pipeline_stream_auth_and_status(monkeypatch):
    monkeypatch.setenv("CAMERA_INTERNAL_TOKEN", "test-token-conn")
    from camera_tracking.streaming.mjpeg import MjpegStreamer
    from camera_tracking.api.access import signed_stream_url
    streamer = MjpegStreamer(port=0)
    assert streamer.start()
    try:
        port = streamer._server.server_address[1]
        streamer.push("cam_a", b"\xff\xd8fakejpeg\xff\xd9")

        def get(path, token=""):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}",
                headers={"Authorization": "Bearer " + token} if token else {})
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    return response.status, json.load(response)
            except Exception as error:
                from urllib.error import HTTPError
                if isinstance(error, HTTPError):
                    return error.code, {}
                raise

        code, _ = get("/status.json")
        assert code == 403  # token required, like production
        code, payload = get("/status.json", "test-token-conn")
        assert code == 200 and payload["cameras"]

        signed = signed_stream_url("http://127.0.0.1:9999", "/cam_a.mjpg")
        assert "signature=" in signed and "expires=" in signed
    finally:
        streamer.stop()


def test_camera_inventory_matches_runtime_ports():
    import yaml
    inventory = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config/cameras.yaml").read_text(encoding="utf-8"))
    assert inventory["network"]["backend_api"].endswith(":8767")
    assert inventory["network"]["stream"].endswith(":8765")
    assert {camera["id"] for camera in inventory["cameras"]} == {"A", "B"}
    assert inventory["network"]["ui"].endswith(":8501")
