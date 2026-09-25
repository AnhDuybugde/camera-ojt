"""Backend-owned business services, durable ingestion and cloud synchronization."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import threading

from backend.app.api.server import make_server
from backend.app.api.service import ApplicationAPI
from backend.app.api.supabase_auth import SupabaseAuth
from camera_tracking.store.aimind_bridge import drain_once, load_person_map
from camera_tracking.store.event_log import EventSync
from camera_tracking.store.queue import WriteQueue

ROOT = Path(__file__).resolve().parents[2]


def load_services():
    # The presentation app's business modules remain import-compatible during
    # the transition. Only this backend process constructs their services.
    sys.path.insert(0, str(ROOT / "apps" / "attendance"))
    from backend.app.services.attendance_service import AttendanceService
    from backend.app.services.attendance_admin_service import AttendanceAdminService
    from auth.service import AuthService
    from backend.app.database.database import Database
    from google_sheets.sheets_client import SheetsClient
    from google_sheets.sync_service import SyncService

    db = Database(os.getenv("CAMERA_DATABASE_PATH") or None)
    return db, AuthService(db), AttendanceService(db), AttendanceAdminService(db), SyncService(db, SheetsClient())


def run_backend(stop=None, port=8767):
    stop = stop or threading.Event()
    db, auth, attendance, admin, sheets = load_services()
    from backend.app.api.operations import Operations
    from camera_tracking.config import load_config
    embedding_path = db.path
    from backend.app.api.enrollment import Enrollment
    from backend.app.recognition.face_detector import FaceDetector
    identity = SupabaseAuth() if os.getenv("SUPABASE_URL") else auth
    api = ApplicationAPI(db, identity, admin, sheets, Operations(db, attendance, embedding_path),
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
                    allowed = (None if os.getenv("CAMERA_SYNC_BIOMETRIC", "0") == "1"
                               else TABLES.keys())
                    replica.sync(transport, allowed_kinds=allowed)
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
