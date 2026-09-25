"""Live MJPEG streaming for the web dashboard (stdlib only, no new deps).

The pipeline pushes annotated JPEG frames per channel; browsers show them
as plain <img> tags (no CORS needed for <img>). /status.json carries the
Global-ID labels and is served with CORS `*` for dashboard polling.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MjpegStreamer:
    """Thread-safe MJPEG server. Call start() once, push() per frame."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._access_key = os.getenv("CAMERA_INTERNAL_TOKEN", "")
        if host not in {"127.0.0.1", "localhost", "::1"} and not self._access_key:
            raise ValueError("CAMERA_INTERNAL_TOKEN is required to expose the camera API on a network")
        self.host = host
        self.port = port
        self._frames: dict[str, bytes] = {}
        self._frame_at: dict[str, float] = {}
        self._status: dict = {"cameras": [], "people": []}
        self._pending_status: dict | None = None
        self._status_interval_s = 0.5
        self._status_published_at = 0.0
        self._attendance_pending: Callable[[], list[dict]] = list
        self._attendance_send: Callable[[], dict] = lambda: {"sent": 0}
        self._enrollment_register: Callable[[dict], dict] = lambda _payload: {
            "ok": False,
            "status": 503,
            "message": "Enrollment is unavailable.",
        }
        self._assistant_ask: Callable[[dict], dict] = lambda _payload: {
            "ok": False,
            "status": 503,
            "message": "Trợ lý Hà Linh chưa sẵn sàng.",
        }
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._server is not None

    def push(self, name: str, jpeg_bytes: bytes) -> None:
        if not jpeg_bytes:
            return
        with self._lock:
            self._frames[name] = bytes(jpeg_bytes)
            self._frame_at[name] = time.monotonic()

    def set_status(self, payload: dict) -> None:
        with self._lock:
            self._pending_status = dict(payload)
            self._publish_status_locked(time.monotonic())

    def _publish_status_locked(self, now: float) -> None:
        if (self._pending_status is not None and
                now - self._status_published_at >= self._status_interval_s):
            self._status = self._pending_status
            self._pending_status = None
            self._status_published_at = now

    def set_attendance_actions(
        self,
        pending: Callable[[], list[dict]],
        send: Callable[[], dict],
    ) -> None:
        self._attendance_pending = pending
        self._attendance_send = send

    def set_enrollment_action(self, register: Callable[[dict], dict]) -> None:
        self._enrollment_register = register

    def set_assistant_action(self, ask: Callable[[dict], dict]) -> None:
        """Register the dashboard text-assistant handler.

        API keys and provider calls remain in the camera backend; dashboards
        only send a question and receive the synthesized answer.
        """
        self._assistant_ask = ask

    def snapshot(self) -> tuple[dict[str, bytes], dict]:
        with self._lock:
            now = time.monotonic()
            self._publish_status_locked(now)
            live = {name: frame for name, frame in self._frames.items()
                    if now - self._frame_at.get(name, 0) < 5.0}
            return live, dict(self._status)

    def start(self) -> bool:
        """Start serving in a daemon thread. False when the port is busy."""
        if self._server is not None:
            return True
        streamer = self

        class _Handler(BaseHTTPRequestHandler):
            server_version = "CamMJPEG/1.0"

            def log_message(self, *args) -> None:  # keep pipeline logs clean
                pass

            def _send_cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", os.getenv("CAMERA_UI_ORIGIN", "http://localhost:8501"))

            def _authorized(self):
                from backend.app.api.access import authorized
                if authorized(self.path, self.headers.get("Authorization", ""), streamer._access_key):
                    return True
                self._send_json({"ok": False, "message": "Authentication required"}, 403)
                return False

            def _send_json(self, payload: dict, status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self._send_cors()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:
                self.send_response(204)
                self._send_cors()
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()

            def do_POST(self) -> None:
                if not self._authorized():
                    return
                path = self.path.split("?", 1)[0]
                if path not in ("/enrollment/register", "/assistant/ask"):
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    content_length = 0
                max_bytes = 8 * 1024 * 1024 if path == "/enrollment/register" else 16 * 1024
                if content_length <= 0 or content_length > max_bytes:
                    self._send_json({
                        "ok": False,
                        "message": "Dữ liệu ảnh không hợp lệ hoặc vượt quá 8 MB.",
                    }, 413)
                    return
                try:
                    payload = json.loads(self.rfile.read(content_length))
                    if not isinstance(payload, dict):
                        raise TypeError("payload must be an object")
                    action = (streamer._enrollment_register
                              if path == "/enrollment/register"
                              else streamer._assistant_ask)
                    result = dict(action(payload))
                    status = int(result.pop("status", 201 if result.get("ok") else 400))
                    self._send_json(result, status)
                except (json.JSONDecodeError, TypeError, UnicodeDecodeError, ValueError):
                    self._send_json({
                        "ok": False,
                        "message": "Dữ liệu đăng ký không hợp lệ.",
                    }, 400)
                except Exception:  # noqa: BLE001 - keep HTTP worker alive
                    self._send_json({
                        "ok": False,
                        "message": "Không thể hoàn tất đăng ký.",
                    }, 500)

            def do_GET(self) -> None:
                if not self._authorized():
                    return
                path = self.path.split("?", 1)[0]
                if path in ("/cam_a.mjpg", "/cam_b.mjpg"):
                    name = "cam_a" if "cam_a" in path else "cam_b"
                    self._serve_mjpeg(streamer, name)
                elif path in ("/status.json", "/status"):
                    frames, status = streamer.snapshot()
                    self._send_json({
                        **status,
                        "cameras": [
                            {"name": n, "live": n in frames}
                            for n in ("cam_a", "cam_b")
                        ],
                    })
                elif path == "/attendance/pending":
                    self._send_json({
                        "pending": streamer._attendance_pending(),
                    })
                elif path == "/attendance/send":
                    result = streamer._attendance_send()
                    self._send_json(result)
                elif path == "/":
                    body = (
                        b"<html><body><h3>Camera OJT live</h3>"
                        b"<img src='/cam_a.mjpg' width='640'/><br/>"
                        b"<img src='/cam_b.mjpg' width='640'/></body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

            def _serve_mjpeg(self, streamer: MjpegStreamer, name: str) -> None:
                # Wait briefly for the first frame so <img> does not 404.
                frame: bytes | None = None
                for _ in range(100):
                    frames, _ = streamer.snapshot()
                    frame = frames.get(name)
                    if frame:
                        break
                    time.sleep(0.1)
                if not frame:
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                try:
                    while True:
                        frames, _ = streamer.snapshot()
                        frame = frames.get(name)
                        if frame:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                                + frame + b"\r\n"
                            )
                        time.sleep(0.04)  # ~25 fps cap; snapshot is always latest.
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        try:
            server = ThreadingHTTPServer((self.host, self.port), _Handler)
            server.daemon_threads = True
        except OSError:
            return False
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever,
                                        kwargs={"poll_interval": 0.2},
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
