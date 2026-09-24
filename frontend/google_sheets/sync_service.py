"""Retryable database-to-Sheets reporting mirror."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from database.db import Database
from auth.permissions import SYSTEM, require_permission
from google_sheets.sheets_client import SheetsClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SyncResult:
    synced: int
    failed: int
    message: str


class SyncService:
    def __init__(self, db: Database, client: SheetsClient) -> None:
        self.db, self.client = db, client
        self._lock = threading.Lock()

    def sync_pending(self, *, actor_role: str, limit: int = 200) -> SyncResult:
        require_permission(actor_role, "sync.manage")
        with self._lock:
            return self._sync_pending(limit)

    def _sync_pending(self, limit: int) -> SyncResult:
        pending = self.db.pending_attendance(limit)
        if not pending:
            return SyncResult(0, 0, "No pending records.")
        if not self.client.configured:
            return SyncResult(0, len(pending), "Google Sheets is not configured; records remain pending.")
        synced, failed = 0, 0
        for record in pending:
            try:
                self.client.upsert_attendance(record)
                self.db.mark_synced(record["id"], record["updated_at"])
                synced += 1
                logger.info("Google Sheets synced attendance id=%s", record["id"])
            except Exception as exc:
                failed += 1
                self.db.mark_sync_error(record["id"], record["updated_at"])
                logger.warning("Google Sheets sync failed for id=%s: %s", record["id"], exc)
        return SyncResult(synced, failed, f"Synced {synced}; failed {failed}.")


class SyncWorker:
    """Daemon that retries pending rows without blocking recognition or UI."""

    def __init__(self, service: SyncService, interval_seconds: int = 30) -> None:
        self.service = service
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "SyncWorker":
        if not self._thread or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="sheets-sync")
            self._thread.start()
        return self

    def request_sync(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.service.sync_pending(actor_role=SYSTEM)
            except Exception:
                logger.exception("Unexpected background sync error")
            self._wake.wait(self.interval_seconds)
            self._wake.clear()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
