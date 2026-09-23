"""Back up local SQLite databases before applying additive application migrations."""
from datetime import datetime
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    target = ROOT / "var/backups" / datetime.now().strftime("%Y%m%d-%H%M%S")
    target.mkdir(parents=True, exist_ok=False)
    count = 0
    for name, source in (
        ("attendance.db", ROOT / "apps/attendance/data/database.db"),
        ("queue.db", ROOT / "output/queue.db"),
        ("enrollment.db", ROOT / "var/enrollment.db"),
        ("identity_state.db", ROOT / "output/identity_state.db"),
    ):
        if source.is_file():
            with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as old, sqlite3.connect(target / name) as new:
                old.backup(new)
            count += 1
    from camera_tracking.application.backend import load_services
    db, *_ = load_services()
    print(f"Backed up {count} databases to {target}; additive migrations applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
