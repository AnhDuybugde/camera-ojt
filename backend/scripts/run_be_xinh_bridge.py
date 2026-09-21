"""Run Be Xinh against the status API exposed by ``run_workstate.py``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
load_dotenv(_PROJECT_ROOT / ".env")

from camera_tracking.audio import CameraCheckInAnnouncer, HamyCompanion  # noqa: E402
from camera_tracking.integration import BackendStatusClient, BeXinhStatusBridge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bé Xinh audio companion for the Camera OJT status API."
    )
    parser.add_argument(
        "--status-url",
        default=os.getenv("TRACKING_STATUS_URL", "http://127.0.0.1:8765/status.json"),
    )
    parser.add_argument("--poll-seconds", type=float, default=0.35)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    announcer = CameraCheckInAnnouncer.from_env()
    if announcer is None:
        print(
            "Bé Xinh chưa thể khởi động. Kiểm tra IMOU_IP, IMOU_USER, "
            "IMOU_PASSWORD và IMOU_TALK_HELPER trong backend/.env.",
            file=sys.stderr,
        )
        return 2

    companion = HamyCompanion.from_env(announcer)
    bridge = BeXinhStatusBridge(companion)
    client = BackendStatusClient(args.status_url, timeout_s=args.timeout_seconds)
    interval_s = max(0.10, float(args.poll_seconds))
    last_error = ""
    last_error_at = float("-inf")

    print(f"Bé Xinh bridge: ACTIVE | {args.status_url}")
    try:
        while True:
            started = time.monotonic()
            try:
                bridge.process(client.fetch(), now_s=started)
                last_error = ""
            except (HTTPError, URLError, OSError, RuntimeError, TypeError, ValueError) as error:
                message = str(error)
                if message != last_error or started - last_error_at >= 10.0:
                    print(f"[Bé Xinh/Bridge] chờ backend: {message}", file=sys.stderr)
                    last_error = message
                    last_error_at = started
            elapsed = time.monotonic() - started
            time.sleep(max(0.02, interval_s - elapsed))
    except KeyboardInterrupt:
        print("Bé Xinh bridge: stopping")
    finally:
        announcer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
