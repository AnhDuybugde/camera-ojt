"""Rooms domain router — backend health, sync conflicts and AI status.

Typed wrapper over the ``operations`` API service. Transport-agnostic: pass
any ``call(service, method, *args, **kwargs)`` — see :func:`connect`.
"""
from __future__ import annotations

from collections.abc import Callable

from backend.app.api._http import connect


class RoomsAPI:
    def __init__(self, call: Callable):
        self._call = call

    @classmethod
    def remote(cls, base_url: str, token: str = "") -> "RoomsAPI":
        return cls(connect(base_url, token))

    def diagnostics(self) -> dict:
        return self._call("operations", "diagnostics")

    def ai_status(self) -> dict:
        return self._call("operations", "ai_status")

    def conflicts(self, store: str = "business") -> list:
        return self._call("operations", "conflicts", store)

    def resolve_conflict(self, kind: str, record_key: str,
                         choice: str, store: str = "business") -> None:
        return self._call("operations", "resolve_conflict",
                          kind, record_key, choice, store)

    def status_summary(self) -> dict:
        """One card for the AI System page: pending work + conflicts."""
        health = self.diagnostics()
        return {
            "pending_events": health.get("pending_events", 0),
            "pending_records": health.get("pending_records", 0),
            "conflicts": health.get("conflicts", 0),
            "review_events": health.get("review_events", 0),
            "last_sync": health.get("last_sync"),
            "sync_error": health.get("sync_error"),
        }
