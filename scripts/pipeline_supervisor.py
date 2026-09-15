"""Local supervisor: lets the Vite dashboard start/stop the camera pipeline.

Browsers cannot spawn local processes, so this tiny stdlib-only HTTP
service (default http://127.0.0.1:8766) owns the pipeline subprocess:

  GET  /api/status -> {running, pid, uptime_s, stream_port, args, log_tail}
  POST /api/start  -> {script, stream_port, imgsz, no_face, display, ...}
  POST /api/stop   -> terminates the pipeline

Run once per boot (or double-click start-all.bat), then use the dashboard
"Pipeline" card. Binds localhost only: anyone with local access can
control the cameras, which is intended for the demo PC.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SCRIPT = "scripts/run_workstate.py"


class PipelineSupervisor:
    def __init__(self, root: Path = _PROJECT_ROOT) -> None:
        self.root = root
        self._proc: subprocess.Popen | None = None
        self._started_at: float | None = None
        self._args: list[str] = []
        self._stream_port: int | None = None
        self._logs: deque[str] = deque(maxlen=200)
        self._lock = threading.Lock()

    # -- process management --
    def _resolve_script(self, script: str) -> Path:
        path = (self.root / script).resolve()
        if self.root.resolve() not in path.parents and path != self.root.resolve():
            raise ValueError("script must stay inside the project folder")
        if not path.is_file():
            raise ValueError(f"script not found: {script}")
        return path

    def start(self, script: str = _DEFAULT_SCRIPT, extra_args: list[str] | None = None,
              stream_port: int | None = None) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return {"started": False, "reason": "already running",
                        "pid": self._proc.pid}
            path = self._resolve_script(script)
            cmd = [sys.executable, str(path), *(extra_args or [])]
            proc = subprocess.Popen(
                cmd, cwd=str(self.root),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            self._proc = proc
            self._started_at = time.time()
            self._args = cmd[2:]
            self._stream_port = stream_port
            self._logs.append(f"$ {' '.join(cmd)}")
            threading.Thread(target=self._drain, args=(proc,), daemon=True).start()
            return {"started": True, "pid": proc.pid}

    def _drain(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                with self._lock:
                    self._logs.append(line.rstrip("\n"))
        except (OSError, ValueError) as error:
            with self._lock:
                self._logs.append(f"[supervisor] log reader stopped: {error}")

    def stop(self) -> dict:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._proc = None
                return {"stopped": False, "reason": "not running"}
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        with self._lock:
            self._proc = None
            self._started_at = None
            self._logs.append("[supervisor] pipeline stopped")
            return {"stopped": True}

    def status(self) -> dict:
        with self._lock:
            proc = self._proc
            running = proc is not None and proc.poll() is None
            if not running:
                self._proc = None
            return {
                "running": running,
                "pid": proc.pid if running and proc else None,
                "uptime_s": round(time.time() - self._started_at, 1)
                if running and self._started_at else 0,
                "stream_port": self._stream_port,
                "args": list(self._args),
                "log_tail": list(self._logs)[-60:],
            }


def _build_start_args(body: dict) -> tuple[str, list[str], int | None]:
    """Whitelist dashboard options into run_workstate CLI args."""
    script = str(body.get("script") or _DEFAULT_SCRIPT)
    stream_port = body.get("stream_port", 8765)
    try:
        stream_port = int(stream_port)
    except (TypeError, ValueError):
        stream_port = 8765
    args: list[str] = []
    if stream_port:
        args += ["--stream-port", str(stream_port)]
    else:
        args += ["--no-stream"]
    imgsz = body.get("imgsz")
    if imgsz:
        try:
            args += ["--imgsz", str(int(imgsz))]
        except (TypeError, ValueError):
            pass
    if body.get("no_face"):
        args.append("--no-face")
    if body.get("display"):
        args.append("--display")
    return script, args, (stream_port or None)


def serve(host: str = "127.0.0.1", port: int = 8766) -> ThreadingHTTPServer:
    supervisor = PipelineSupervisor()

    class _Handler(BaseHTTPRequestHandler):
        server_version = "CamSupervisor/1.0"

        def log_message(self, *args) -> None:
            pass

        def _json(self, payload: dict, code: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self._json({})

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/api/status", "/api/health"):
                self._json(supervisor.status())
            elif path == "/":
                body = (b"<html><body><h3>Pipeline supervisor</h3>"
                        b"<pre id=s></pre><script>fetch('/api/status')"
                        b".then(r=>r.json()).then(j=>s.textContent="
                        b"JSON.stringify(j,null,2))</script></body></html>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, OSError):
                body = {}
            try:
                if path == "/api/start":
                    script, args, stream_port = _build_start_args(body)
                    self._json(supervisor.start(script, args, stream_port))
                elif path == "/api/stop":
                    self._json(supervisor.stop())
                else:
                    self._json({"error": "not found"}, 404)
            except ValueError as error:
                self._json({"error": str(error)}, 400)

    server = ThreadingHTTPServer((host, port), _Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera pipeline supervisor.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    server = serve(args.host, args.port)
    print(f"Supervisor: http://{args.host}:{args.port} (dashboard Pipeline card)")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
