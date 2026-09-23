"""Bridge consumer: camera-ojt WriteQueue -> ai-mind-attendance SQLite.

Reads ``aimind_tick`` rows (CHECK_IN on face tick, CHECK_OUT on
fusion just_left_office) and delivers each through
ai-mind-attendance's AttendanceService.record(), which owns all
schedule / cooldown / checkout rules. Google Sheets sync is untouched
(record() marks rows PENDING for the existing SyncWorker).

Run alongside the pipeline (terminal 2):

  python scripts/bridge_aimind_attendance.py
  python scripts/bridge_aimind_attendance.py --once   # drain + exit
  python scripts/bridge_aimind_attendance.py --poll 2 --verbose

Mapping gallery person_id -> employee_id lives in
ai-mind-attendance/data/person_map.json
({"AnhDuy": "NV001", ...}, override with --map).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_AIMIND_ROOT = _PROJECT_ROOT / "ai-mind-attendance"
sys.path.insert(0, str(_AIMIND_ROOT))

from camera_tracking.store.aimind_bridge import drain_once, load_person_map
from camera_tracking.store.queue import WriteQueue

log = logging.getLogger("aimind-bridge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drain camera-ojt aimind_tick rows into ai-mind-attendance.")
    parser.add_argument("--queue", type=Path,
                        default=_PROJECT_ROOT / "output" / "queue.db")
    parser.add_argument("--map", type=Path,
                        default=_AIMIND_ROOT / "data" / "person_map.json")
    parser.add_argument("--db", type=Path,
                        default=_AIMIND_ROOT / "data" / "database.db")
    parser.add_argument("--poll", type=float, default=5.0,
                        help="Seconds between drains (default 5).")
    parser.add_argument("--once", action="store_true",
                        help="Drain once and exit (cron/systemd use).")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    from attendance.attendance_service import AttendanceService
    from database.db import Database

    person_map = load_person_map(args.map)
    log.info("Person map: %d entries from %s", len(person_map), args.map)
    queue = WriteQueue(args.queue)
    service = AttendanceService(Database(args.db))
    while True:
        try:
            stats = drain_once(queue, service, person_map)
        except (OSError, ValueError) as error:
            log.warning("Drain failed, retry in %.1fs: %s", args.poll, error)
            stats = {"processed": 0, "errors": 1}
        if stats.get("processed"):
            log.info("Drain: %s", stats)
        elif args.verbose:
            log.debug("Drain: queue empty")
        if args.once:
            return 0 if not stats.get("errors") else 1
        time.sleep(max(0.5, args.poll))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
