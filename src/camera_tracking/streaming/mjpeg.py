"""Live MJPEG streaming for the web dashboard (stdlib only, no new deps).

The pipeline pushes annotated JPEG frames per channel; browsers show them
as plain <img> tags (no CORS needed for <img>). /status.json carries the
Global-ID labels and is served with CORS `*` for dashboard polling.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MjpegStreamer:
    """Thread-safe MJPEG server. Call start() once, push() per frame."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._frames: dict[str, bytes] = {}
        self._status: dict = {"cameras": [], "people": []}
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

    def set_status(self, payload: dict) -> None:
        with self._lock:
            self._status = dict(payload)

    def snapshot(self) -> tuple[dict[str, bytes], dict]:
        with self._lock:
            return dict(self._frames), dict(self._status)

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

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path in ("/cam_a.mjpg", "/cam_b.mjpg"):
                    name = "cam_a" if "cam_a" in path else "cam_b"
                    self._serve_mjpeg(streamer, name)
                elif path in ("/status.json", "/status"):
                    frames, status = streamer.snapshot()
                    body = json.dumps({
                        **status,
                        "cameras": [
                            {"name": n, "live": n in frames}
                            for n in ("cam_a", "cam_b")
                        ],
                    }).encode("utf-8")
                    self.send_response(200)
                    self._send_cors()
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
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
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def _serve_mjpeg(self, streamer: "MjpegStreamer", name: str) -> None:
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
                        time.sleep(0.07)  # ~14 fps cap per viewer
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
