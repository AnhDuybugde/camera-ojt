"""Administration operations backed by the same database as attendance."""
from datetime import datetime
import json
from uuid import uuid5, NAMESPACE_URL

from camera_tracking.store.replication import SQLiteReplica, TABLES, ENROLLMENT_TABLES


class Operations:
    def __init__(self, db, attendance, embedding_path):
        self.db, self.attendance, self.embedding_path = db, attendance, embedding_path

    def diagnostics(self):
        def scalar(sql):
            return next(iter(self.db.fetch_one(sql).values()))
        return {
            "pending_events": scalar("SELECT COUNT(*) FROM event_outbox WHERE delivered_at IS NULL"),
            "pending_records": scalar("SELECT COUNT(*) FROM record_outbox"),
            "conflicts": scalar("SELECT COUNT(*) FROM record_conflicts"),
            "review_events": len(self.review_events()),
            "last_sync": scalar("SELECT MAX(delivered_at) FROM event_outbox"),
            "sync_error": scalar("SELECT last_error FROM event_outbox WHERE last_error IS NOT NULL LIMIT 1")
                if self.db.fetch_one("SELECT 1 FROM event_outbox WHERE last_error IS NOT NULL LIMIT 1") else None,
        }

    def review_events(self):
        return self.db.fetch_all("""SELECT e.* FROM camera_events e WHERE e.needs_review=1
            AND NOT EXISTS (SELECT 1 FROM camera_events r
                WHERE json_extract(r.payload,'$.review_of')=e.event_id)
            ORDER BY e.observed_at DESC LIMIT 200""")

    def review(self, event_id, employee_id, reason):
        if len(reason.strip()) < 5:
            raise ValueError("A review reason is required")
        event = self.db.fetch_one("SELECT * FROM camera_events WHERE event_id=?", (event_id,))
        if not event or event["kind"] not in {"CHECK_IN", "CHECK_OUT"}:
            raise ValueError("Event is not reviewable")
        result = self.attendance.record_direction(employee_id,
            datetime.fromisoformat(event["observed_at"]), direction=event["kind"],
            event_id=str(uuid5(NAMESPACE_URL, "camera-review:" + event_id)),
            camera_id=event["camera_id"], session_id=event["session_id"],
            evidence={"review_of": event_id, "reason": reason.strip(), "actor": "ADMIN"})
        return {"action": result.action, "message": result.message}

    def conflicts(self, store="business"):
        replica = self._replica(store)
        with replica.connect() as db:
            return [dict(row) for row in db.execute("SELECT kind,record_key,remote FROM record_conflicts")]

    def resolve_conflict(self, kind, record_key, choice, store="business"):
        self._replica(store).resolve(kind, record_key, choice)
        self.db.execute("INSERT INTO audit_logs(timestamp,role,action,new_values,reason) VALUES (?,?,?,?,?)",
            (datetime.now().astimezone().isoformat(), "ADMIN", "SYNC_CONFLICT_RESOLVED",
             json.dumps({"kind": kind, "key": record_key, "store": store}), choice))

    def _replica(self, store):
        if store not in {"business", "enrollment"}:
            raise ValueError("Unknown replica")
        return SQLiteReplica(self.db.path, {**TABLES, **ENROLLMENT_TABLES})
