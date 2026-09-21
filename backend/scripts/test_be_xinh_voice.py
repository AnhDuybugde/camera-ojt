"""One-shot end-to-end smoke test for the Bé Xinh camera speaker."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

from camera_tracking.audio import CameraCheckInAnnouncer  # noqa: E402


DEFAULT_TEXT = "Bé Xinh đã kết nối thành công. Xin chào Quốc Ngọc."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--synthesis-timeout", type=float, default=900.0)
    parser.add_argument("--playback-timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    speaker = CameraCheckInAnnouncer.from_env()
    if speaker is None:
        print("FAILED: kiểm tra IMOU_IP/USER/PASSWORD và IMOU_TALK_HELPER.")
        return 2

    try:
        if not speaker.is_cached(args.text):
            print("CACHE: chưa có, đang tạo một câu kiểm tra...")
            if speaker.prewarm([args.text]) != 1:
                print("FAILED: không thể đưa câu kiểm tra vào hàng đợi cache.")
                return 3
            if not speaker.wait_idle(timeout_s=args.synthesis_timeout):
                print("FAILED: hết thời gian tạo cache.")
                return 4
            if not speaker.is_cached(args.text):
                print("FAILED: TTS không tạo được cache.")
                return 5
            print("CACHE: READY")
        else:
            print("CACHE: READY (đã có)")

        print("PLAYBACK: đang phát qua loa camera...")
        if not speaker.say(args.text, expires_s=60.0):
            print("FAILED: không thể đưa câu kiểm tra vào hàng đợi phát.")
            return 6
        if not speaker.wait_idle(timeout_s=args.playback_timeout):
            print("FAILED: hết thời gian chờ loa camera.")
            return 7
        print("PLAYBACK: SUCCESS")
        return 0
    finally:
        speaker.close()


if __name__ == "__main__":
    raise SystemExit(main())
