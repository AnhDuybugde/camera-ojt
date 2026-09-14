"""Live MJPEG streaming for the web dashboard (stdlib only, no new deps).

The pipeline pushes annotated JPEG frames per channel; browsers show them
as plain <img> tags (no CORS needed for <img>). /status.json carries the
Global-ID labels and is served with CORS `*` for dashboard polling.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote


class MjpegStreamer:
    """Thread-safe MJPEG server. Call start() once, push() per frame."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._frames: dict[str, bytes] = {}
        self._frame_times: dict[str, float] = {}
        self._status: dict = {"cameras": [], "people": []}
        self._attendance_pending: Callable[[], list[dict]] = list
        self._attendance_send: Callable[[], dict] = lambda: {"sent": 0}
        self._enrollment_register: Callable[[dict], dict] = lambda _payload: {
            "ok": False,
            "status": 503,
            "message": "Enrollment is unavailable.",
        }
        self._enrollment_list: Callable[[], list[dict]] = list
        self._enrollment_delete: Callable[[str], dict] = lambda _person_id: {
            "ok": False,
            "status": 503,
            "message": "Enrollment management is unavailable.",
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
            self._frame_times[name] = time.monotonic()

    def set_status(self, payload: dict) -> None:
        with self._lock:
            self._status = dict(payload)

    def set_attendance_actions(
        self,
        pending: Callable[[], list[dict]],
        send: Callable[[], dict],
    ) -> None:
        self._attendance_pending = pending
        self._attendance_send = send

    def set_enrollment_actions(
        self,
        register: Callable[[dict], dict],
        list_people: Callable[[], list[dict]],
        delete: Callable[[str], dict],
    ) -> None:
        self._enrollment_register = register
        self._enrollment_list = list_people
        self._enrollment_delete = delete

    def snapshot(self) -> tuple[dict[str, bytes], dict]:
        with self._lock:
            return dict(self._frames), dict(self._status)

    def camera_live(self, name: str, max_age_s: float = 3.0) -> bool:
        with self._lock:
            last_frame_s = self._frame_times.get(name)
        return bool(
            last_frame_s is not None
            and time.monotonic() - last_frame_s <= max_age_s
        )

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
                self.send_header("Access-Control-Allow-Origin", "*")

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
                self.send_header(
                    "Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"
                )
                self.end_headers()

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                if path != "/enrollment/register":
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    content_length = 0
                if content_length <= 0 or content_length > 8 * 1024 * 1024:
                    self._send_json({
                        "ok": False,
                        "message": "Dữ liệu ảnh không hợp lệ hoặc vượt quá 8 MB.",
                    }, 413)
                    return
                try:
                    payload = json.loads(self.rfile.read(content_length))
                    if not isinstance(payload, dict):
                        raise TypeError("payload must be an object")
                    result = dict(streamer._enrollment_register(payload))
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

            def do_DELETE(self) -> None:
                path = self.path.split("?", 1)[0]
                prefix = "/enrollment/persons/"
                if not path.startswith(prefix):
                    self.send_response(404)
                    self.end_headers()
                    return
                person_id = unquote(path[len(prefix):]).strip()
                if not person_id or "/" in person_id or "\\" in person_id:
                    self._send_json({
                        "ok": False,
                        "message": "Mã nhân viên không hợp lệ.",
                    }, 400)
                    return
                try:
                    result = dict(streamer._enrollment_delete(person_id))
                    status = int(result.pop("status", 200 if result.get("ok") else 400))
                    self._send_json(result, status)
                except Exception:  # noqa: BLE001 - keep HTTP worker alive
                    self._send_json({
                        "ok": False,
                        "message": "Không thể xóa hồ sơ nhận diện.",
                    }, 500)

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path in ("/cam_a.mjpg", "/cam_b.mjpg"):
                    name = "cam_a" if "cam_a" in path else "cam_b"
                    self._serve_mjpeg(streamer, name)
                elif path in ("/status.json", "/status"):
                    _frames, status = streamer.snapshot()
                    self._send_json({
                        **status,
                        "cameras": [
                            {"name": n, "live": streamer.camera_live(n)}
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
                elif path == "/enrollment/persons":
                    self._send_json({"people": streamer._enrollment_list()})
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
