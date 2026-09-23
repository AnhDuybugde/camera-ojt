"""Drain versioned local data to Supabase after migration; never prints payloads."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from camera_tracking.application.backend import load_services
    from camera_tracking.store.cloud import CloudEventWriter
    from camera_tracking.store.replication import SQLiteReplica, TABLES, ENROLLMENT_TABLES
    from camera_tracking.store.event_log import EventSync
    db, *_ = load_services()
    replica = SQLiteReplica(db.path, {**TABLES, **ENROLLMENT_TABLES})
    remote = CloudEventWriter.from_env()
    replica.bootstrap()
    for _ in range(30):
        replica.sync(remote)
        with replica.connect() as connection:
            pending = connection.execute("SELECT count(*) FROM record_outbox").fetchone()[0]
            conflicts = connection.execute("SELECT count(*) FROM record_conflicts").fetchone()[0]
            cursor = connection.execute("SELECT sequence FROM sync_cursor WHERE id=1").fetchone()[0]
        if pending == conflicts and len(remote.changes(cursor)) == 0:
            break
    print(replica.path.name, "pending:", pending, "conflicts:", conflicts)
    print("Events delivered:", EventSync(db, remote).flush())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("Synchronization deferred:", type(error).__name__)
        raise SystemExit(1)
