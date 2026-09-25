"""Compatibility shim — canonical home is backend.app.database.database."""
from backend.app.database.database import WORK_SESSIONS, WORK_STATUSES, Database

__all__ = ["WORK_SESSIONS", "WORK_STATUSES", "Database"]
