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


def doctor():
    import shutil
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    result = {"python": sys.version.split()[0], "ffmpeg": bool(shutil.which("ffmpeg")),
              "modules": {}, "configured": {}}
    for name in ("cv2", "numpy", "insightface", "onnxruntime", "torch", "streamlit", "bcrypt"):
        result["modules"][name] = importlib.util.find_spec(name) is not None
    for name in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY"):
        result["configured"][name] = bool(os.getenv(name))
    gpu = shutil.which("nvidia-smi")
    result["gpu_driver_ready"] = bool(gpu and subprocess.run(
        [gpu, "--query-gpu=name", "--format=csv,noheader"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10).returncode == 0)
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
        from camera_tracking.application.backend import run_backend
        run_backend(stop)
        return 0
    children = []
    env = dict(os.environ)
    import secrets
    env.setdefault("CAMERA_INTERNAL_TOKEN", secrets.token_urlsafe(32))
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    try:
        children.append(subprocess.Popen([sys.executable, "-m", "camera_tracking.cli", "backend"], env=env))
        if not args.no_camera:
            script = "run_workstate_local.py" if args.profile == "webcam" else "run_workstate.py"
            children.append(subprocess.Popen([sys.executable, str(ROOT / "scripts" / script),
                                             "--no-supervisor", *extra], env=env))
        if not args.no_ui:
            children.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run",
                str(ROOT / "apps/attendance/app.py"), "--server.address", args.ui_host,
                "--browser.gatherUsageStats", "false", "--server.enableStaticServing", "true"],
                env=env, cwd=ROOT / "apps/attendance"))
        while not stop.wait(0.5):
            for child in children:
                if child.poll() is not None:
                    return child.returncode or 1
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
