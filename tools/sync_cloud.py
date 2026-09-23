"""Drain versioned local data to Supabase after migration; never prints payloads."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-biometric", action="store_true",
                        help="also sync versioned face embeddings; requires data-protection preflight")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from camera_tracking.application.backend import load_services
    from camera_tracking.store.cloud import CloudEventWriter
    from camera_tracking.store.replication import SQLiteReplica, TABLES, ENROLLMENT_TABLES
    from camera_tracking.store.event_log import EventSync
    db, *_ = load_services()
    replica = SQLiteReplica(db.path, {**TABLES, **ENROLLMENT_TABLES})
    allowed = None if args.include_biometric else TABLES.keys()
    remote = CloudEventWriter.from_env()
    replica.bootstrap()
    for _ in range(30):
        replica.sync(remote, allowed_kinds=allowed)
        with replica.connect() as connection:
            pending = connection.execute("SELECT kind,count(*) FROM record_outbox GROUP BY kind").fetchall()
            conflicts = connection.execute("SELECT kind,count(*) FROM record_conflicts GROUP BY kind").fetchall()
            cursor = connection.execute("SELECT sequence FROM sync_cursor WHERE id=1").fetchone()[0]
        pending_by_kind, conflicts_by_kind = dict(pending), dict(conflicts)
        active = None if allowed is None else set(allowed)
        active_pending = sum(n for kind, n in pending_by_kind.items() if active is None or kind in active)
        active_conflicts = sum(n for kind, n in conflicts_by_kind.items() if active is None or kind in active)
        if active_pending == active_conflicts and len(remote.changes(cursor)) == 0:
            break
    print(replica.path.name, "pending by kind:", pending_by_kind,
          "conflicts by kind:", conflicts_by_kind,
          "embedding sync enabled:", args.include_biometric)
    print("Events delivered:", EventSync(db, remote).flush())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("Synchronization deferred:", type(error).__name__)
        raise SystemExit(1)
