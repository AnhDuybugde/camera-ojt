from __future__ import annotations

import json
import urllib.error
import urllib.request

from camera_tracking.streaming.mjpeg import MjpegStreamer


def _request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict]:
    request = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def test_remote_stream_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("CAMERA_ALLOW_REMOTE_STREAM", raising=False)
    streamer = MjpegStreamer(host="0.0.0.0", port=0)
    assert streamer.start() is False
    assert streamer.running is False


def test_attendance_send_is_post_only() -> None:
    streamer = MjpegStreamer(host="127.0.0.1", port=0)
    streamer.set_attendance_actions(lambda: [], lambda: {"sent": 2, "failed": 0})
    assert streamer.start() is True
    try:
        assert streamer._server is not None  # noqa: SLF001 - integration test
        port = streamer._server.server_port  # noqa: SLF001
        code, body = _request(f"http://127.0.0.1:{port}/attendance/send")
        assert code == 405
        assert body["error"] == "method_not_allowed"

        code, body = _request(
            f"http://127.0.0.1:{port}/attendance/send", method="POST"
        )
        assert code == 200
        assert body == {"sent": 2, "failed": 0}
    finally:
        streamer.stop()


def test_attendance_send_rejects_untrusted_browser_origin() -> None:
    calls = 0

    def send() -> dict:
        nonlocal calls
        calls += 1
        return {"sent": 1, "failed": 0}

    streamer = MjpegStreamer(host="127.0.0.1", port=0)
    streamer.set_attendance_actions(lambda: [], send)
    assert streamer.start() is True
    try:
        assert streamer._server is not None  # noqa: SLF001 - integration test
        port = streamer._server.server_port  # noqa: SLF001
        code, body = _request(
            f"http://127.0.0.1:{port}/attendance/send",
            method="POST",
            headers={"Origin": "https://untrusted.example"},
        )
        assert code == 403
        assert body == {"error": "forbidden_origin"}
        assert calls == 0
    finally:
        streamer.stop()


def test_health_and_readiness_reflect_fresh_pipeline_state() -> None:
    streamer = MjpegStreamer(host="127.0.0.1", port=0)
    assert streamer.start() is True
    try:
        assert streamer._server is not None  # noqa: SLF001 - integration test
        port = streamer._server.server_port  # noqa: SLF001

        code, body = _request(f"http://127.0.0.1:{port}/healthz")
        assert code == 200
        assert body == {"ok": True, "service": "camera-stream"}

        code, body = _request(f"http://127.0.0.1:{port}/readyz")
        assert code == 503
        assert body["ok"] is False
        assert body["status_fresh"] is False

        streamer.push("cam_a", b"jpeg")
        streamer.set_status({"count": 1})
        code, body = _request(f"http://127.0.0.1:{port}/readyz")
        assert code == 200
        assert body["ok"] is True
        assert body["cameras"]["cam_a"]["live"] is True
        assert body["cameras"]["cam_b"]["live"] is False
    finally:
        streamer.stop()
