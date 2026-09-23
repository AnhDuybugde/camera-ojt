"""Backend-owned business services, durable ingestion and cloud synchronization."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import threading

from camera_tracking.api.server import make_server
from camera_tracking.api.service import ApplicationAPI
from camera_tracking.store.aimind_bridge import drain_once, load_person_map
from camera_tracking.store.event_log import EventSync
from camera_tracking.store.queue import WriteQueue

ROOT = Path(__file__).resolve().parents[3]


def load_services():
    # The presentation app's business modules remain import-compatible during
    # the transition. Only this backend process constructs their services.
    sys.path.insert(0, str(ROOT / "apps" / "attendance"))
    from attendance.attendance_service import AttendanceService
    from attendance.admin_service import AttendanceAdminService
    from auth.service import AuthService
    from database.db import Database
    from google_sheets.sheets_client import SheetsClient
    from google_sheets.sync_service import SyncService

    db = Database(os.getenv("CAMERA_DATABASE_PATH") or None)
    return db, AuthService(db), AttendanceService(db), AttendanceAdminService(db), SyncService(db, SheetsClient())


def run_backend(stop=None, port=8767):
    stop = stop or threading.Event()
    db, auth, attendance, admin, sheets = load_services()
    from camera_tracking.api.operations import Operations
    from camera_tracking.config import load_config
    embedding_path = db.path
    from camera_tracking.api.enrollment import Enrollment
    from face.detector import FaceDetector
    api = ApplicationAPI(db, auth, admin, sheets, Operations(db, attendance, embedding_path),
                         Enrollment(FaceDetector()))
    server = make_server(api, port=port)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    queue = WriteQueue(ROOT / "output" / "queue.db")
    person_map = ROOT / "apps" / "attendance" / "data" / "person_map.json"
    cloud = None
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"):
        from camera_tracking.store.cloud import CloudEventWriter
        from camera_tracking.store.replication import SQLiteReplica, TABLES, ENROLLMENT_TABLES
        transport = CloudEventWriter.from_env()
        cloud = EventSync(db, transport)
        replica = SQLiteReplica(db.path, {**TABLES, **ENROLLMENT_TABLES})
        replica.bootstrap()

    def sync_loop():
        while not stop.is_set():
            try:
                if cloud:
                    replica.sync(transport)
                    cloud.flush()
                sheets.sync_pending(actor_role="SYSTEM")
            except Exception as error:
                logging.warning("Cloud sync deferred: %s", type(error).__name__)
            stop.wait(5)

    cloud_thread = threading.Thread(target=sync_loop, name="cloud-sync", daemon=True)
    cloud_thread.start()
    try:
        while not stop.is_set():
            try:
                from camera_tracking.face.gallery import load_registry
                mappings = {key: row["employee_id"] for key, row in
                            load_registry(ROOT / "data/images").items()}
                mappings.update(load_person_map(person_map))
                queue.restore_mapped(mappings)
                drain_once(queue, attendance, mappings)
            except Exception as error:
                logging.warning("Ingestion deferred: %s", type(error).__name__)
            stop.wait(0.5)
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)
        cloud_thread.join(timeout=15)
