"""Dry-run or play one semantic/generic Audio Zone event."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")

from camera_tracking.audio import (  # noqa: E402
    AudioEvent,
    AudioEventKind,
    AudioEventRouter,
    CameraCheckInAnnouncer,
)


class PrintSpeaker:
    enabled = True

    def say(self, text: str, **kwargs) -> bool:
        print(f"[DRY-RUN] {text}")
        print(f"          policy={kwargs}")
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event",
        choices=[kind.value for kind in AudioEventKind],
        default=AudioEventKind.DOOR_ENTER.value,
    )
    parser.add_argument(
        "--generic-zone",
        choices=["be_xinh", "water", "restroom"],
        help="emit ZONE_DWELL instead of the semantic --event",
    )
    parser.add_argument("--dwell-seconds", type=float, default=2.0)
    parser.add_argument("--person-id", default="NV001")
    parser.add_argument("--name", default="Minh")
    parser.add_argument("--gid", type=int, default=1)
    parser.add_argument("--camera", default="A")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    speaker = PrintSpeaker() if args.dry_run else CameraCheckInAnnouncer.from_env()
    if speaker is None:
        print("FAILED: kiểm tra IMOU_IP/USER/PASSWORD và IMOU_TALK_HELPER.")
        return 2
    now_wall = time.time()
    if args.generic_zone:
        event = AudioEvent.from_payload({
            "event_id": f"manual-{int(now_wall * 1000)}-zone-dwell",
            "type": "ZONE_DWELL",
            "zone": args.generic_zone,
            "dwell_seconds": args.dwell_seconds,
            "timestamp": now_wall,
            "global_id": args.gid,
            "person_id": args.person_id,
            "display_name": args.name,
            "camera": args.camera,
        })
        if event is None:
            print("FAILED: generic zone event không hợp lệ.")
            return 2
    else:
        event = AudioEvent(
            event_id=f"manual-{int(now_wall * 1000)}-{args.event}",
            kind=AudioEventKind(args.event),
            timestamp=now_wall,
            source_type=args.event,
            global_id=args.gid,
            person_id=args.person_id,
            display_name=args.name,
            camera=args.camera,
        )

    router = AudioEventRouter(speaker)
    try:
        result = router.route(event, wall_time_s=now_wall)
        print(f"event={result.kind.value} accepted={result.accepted} reason={result.reason}")
        return 0 if result.accepted else 1
    finally:
        close = getattr(speaker, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
