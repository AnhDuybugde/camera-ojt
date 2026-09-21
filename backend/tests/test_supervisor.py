"""Test pipeline supervisor: arg building + start/stop/status over HTTP."""
import importlib.util
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_SUPERVISOR = Path(__file__).resolve().parents[1] / "scripts" / "pipeline_supervisor.py"
_ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("pipeline_supervisor", _SUPERVISOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_build_start_args() -> None:
    mod = _load()
    script, args, port = mod._build_start_args({})
    assert script == "scripts/run_workstate.py"
    assert args == ["--stream-port", "8765"]
    assert port == 8765
    script, args, port = mod._build_start_args(
        {"imgsz": 640, "no_face": True, "display": True, "stream_port": 0})
    assert args == ["--no-stream", "--imgsz", "640", "--no-face", "--display"]
    assert port is None


def test_start_stop_lifecycle() -> None:
    mod = _load()
    staged = _ROOT / "_tmp_sup_dummy.py"
    staged.write_text('import time\nprint("dummy up", flush=True)\ntime.sleep(60)\n')
    port = _free_port()
    server = mod.serve("127.0.0.1", port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"

        def post(path: str, body: dict) -> dict:
            req = urllib.request.Request(
                base + path, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=15) as res:
                return json.loads(res.read())

        def get(path: str) -> dict:
            with urllib.request.urlopen(base + path, timeout=15) as res:
                return json.loads(res.read())

        assert get("/api/status")["running"] is False
        assert post("/api/start", {"script": "_tmp_sup_dummy.py"})["started"] is True
        time.sleep(1.5)
        status = get("/api/status")
        assert status["running"] is True
        assert any("dummy up" in line for line in status["log_tail"])
        assert post("/api/start", {})["started"] is False  # already running
        assert post("/api/stop", {})["stopped"] is True
        assert get("/api/status")["running"] is False
        # Path traversal outside the project is blocked.
        try:
            post("/api/start", {"script": "../../evil.py"})
            raise AssertionError("traversal should be blocked")
        except urllib.error.HTTPError as error:
            assert error.code == 400
    finally:
        server.shutdown()
        staged.unlink(missing_ok=True)


def test_managed_embedded_status() -> None:
    """Embedded mode (run_workstate): running=True, start/stop read-only."""
    mod = _load()
    port = _free_port()
    server = mod.serve(
        "127.0.0.1", port, managed_pid=12345,
        managed_stream_port=8765, managed_args=["--greet"],
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"

        def get(path: str) -> dict:
            with urllib.request.urlopen(base + path, timeout=15) as res:
                return json.loads(res.read())

        def post(path: str, body: dict) -> dict:
            req = urllib.request.Request(
                base + path, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=15) as res:
                return json.loads(res.read())

        status = get("/api/status")
        assert status["running"] is True
        assert status["pid"] == 12345
        assert status["embedded"] is True
        assert status["stream_port"] == 8765
        assert post("/api/start", {})["started"] is False  # already running
        assert post("/api/stop", {})["stopped"] is False  # managed externally
    finally:
        server.shutdown()
