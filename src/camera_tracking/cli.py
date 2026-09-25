"""One entry point for the application, diagnostics and replay."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]


def _port_in_use(host, port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def doctor():
    import shutil
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    result = {"python": sys.version.split()[0], "ffmpeg": bool(shutil.which("ffmpeg")),
              "modules": {}, "configured": {}, "ports": {}, "backend_reachable": False}
    for name in ("cv2", "numpy", "insightface", "onnxruntime", "torch", "streamlit", "bcrypt"):
        result["modules"][name] = importlib.util.find_spec(name) is not None
    for name in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY"):
        result["configured"][name] = bool(os.getenv(name))
    gpu = shutil.which("nvidia-smi")
    result["gpu_driver_ready"] = bool(gpu and subprocess.run(
        [gpu, "--query-gpu=name", "--format=csv,noheader"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10).returncode == 0)
    for label, port in (("backend_8767", 8767), ("stream_8765", 8765),
                        ("supervisor_8766", 8766), ("ui_8501", 8501)):
        result["ports"][label] = "busy" if _port_in_use("127.0.0.1", port) else "free"
    api_url = os.getenv("CAMERA_API_URL", "http://127.0.0.1:8767").rstrip("/")
    try:
        import urllib.error
        from urllib.request import Request, urlopen
        request = Request(f"{api_url}/api/v1/operations/diagnostics",
            data=json.dumps({"args": [], "kwargs": {}}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            urlopen(request, timeout=3)
            result["backend_reachable"] = True
        except urllib.error.HTTPError as error:
            # 403 without token / 400 / 500 still prove the backend listens.
            result["backend_reachable"] = error.code in (400, 403, 500)
    except Exception:
        result["backend_reachable"] = False
    print(json.dumps(result, indent=2))
    return 0 if all(result["modules"].values()) else 1


def main():
    parser = argparse.ArgumentParser(prog="camera-ojt")
    parser.add_argument("command", choices=("run", "backend", "doctor", "replay"))
    parser.add_argument("--profile", choices=("imou", "webcam"), default="imou")
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--ui-host", default="127.0.0.1")
    args, extra = parser.parse_known_args()
    if args.command == "doctor":
        return doctor()
    os.chdir(ROOT)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if args.command == "replay":
        if not any(arg == "--source-a" or arg.startswith("--source-a=") for arg in extra):
            parser.error("replay requires --source-a VIDEO (and optionally --source-b VIDEO)")
        sys.argv = [sys.argv[0], *extra, "--no-halinh", "--no-greet", "--no-supervisor"]
        from camera_tracking.application.workstate import main as pipeline
        pipeline("replay")
        return 0
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    if args.command == "backend":
        from backend.app.main import run_backend
        run_backend(stop)
        return 0
    children = []
    env = dict(os.environ)
    import secrets
    env.setdefault("CAMERA_INTERNAL_TOKEN", secrets.token_urlsafe(32))
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Fail fast with a clear message when a previous run still holds a port.
    wanted_ports = [("backend :8767", 8767)]
    if not args.no_camera:
        wanted_ports += [("stream :8765", 8765), ("supervisor :8766", 8766)]
    if not args.no_ui:
        wanted_ports.append(("UI :8501", 8501))
    busy = [label for label, port in wanted_ports if _port_in_use("127.0.0.1", port)]
    if busy:
        print(f"Lỗi: cổng đã bị chiếm ({', '.join(busy)}). "
              "Tắt tiến trình camera-ojt cũ đang chạy rồi thử lại.", file=sys.stderr)
        return 1
    backend_child = None
    try:
        backend_child = subprocess.Popen([sys.executable, "-m", "camera_tracking.cli", "backend"], env=env)
        children.append(("backend", backend_child))
        if not args.no_camera:
            script = "run_workstate_local.py" if args.profile == "webcam" else "run_workstate.py"
            children.append(("pipeline", subprocess.Popen(
                [sys.executable, str(ROOT / "scripts" / script),
                 "--no-supervisor", *extra], env=env)))
        if not args.no_ui:
            children.append(("ui", subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run",
                 str(ROOT / "apps/attendance/app.py"), "--server.address", args.ui_host,
                 "--browser.gatherUsageStats", "false", "--server.enableStaticServing", "true"],
                env=env, cwd=ROOT / "apps/attendance")))
        # Give the backend a moment to bind :8767 before reporting status.
        for _ in range(20):
            if backend_child.poll() is not None:
                print("Lỗi: backend :8767 thoát ngay khi khởi động. "
                      "Xem log tiến trình backend để biết nguyên nhân.", file=sys.stderr)
                return backend_child.returncode or 1
            if _port_in_use("127.0.0.1", 8767):
                break
            time.sleep(0.25)
        dead_optional = set()
        while not stop.wait(0.5):
            if backend_child.poll() is not None:
                print(f"Lỗi: backend :8767 đã dừng (mã {backend_child.returncode}). "
                      "UI/Pipeline không thể kết nối. Khởi động lại bằng camera-ojt run.",
                      file=sys.stderr)
                return backend_child.returncode or 1
            for name, child in children:
                if name == "backend" or name in dead_optional:
                    continue
                if child.poll() is not None:
                    # Pipeline/UI are optional: keep the backend (and the other
                    # side) alive so the UI still shows backend data instead of
                    # dropping the whole stack on a camera failure.
                    print(f"Cảnh báo: tiến trình {name} đã dừng (mã {child.returncode}). "
                          f"Backend :8767 vẫn chạy; khởi động lại {name} hoặc "
                          "chạy camera-ojt run --no-camera để dùng chế độ quản trị.",
                          file=sys.stderr)
                    dead_optional.add(name)
    finally:
        for _, child in children:
            if child.poll() is None:
                child.terminate()
        for _, child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
