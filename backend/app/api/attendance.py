"""Attendance domain router — records, corrections and review queue.

Typed wrapper over the ``employees`` (read) + ``attendance`` (corrections)
+ ``operations`` (review) API services. Transport-agnostic: pass any
``call(service, method, *args, **kwargs)`` — see :func:`connect`.
"""
from __future__ import annotations

from collections.abc import Callable

from backend.app.api._http import connect


class AttendanceAPI:
    def __init__(self, call: Callable):
        self._call = call

    @classmethod
    def remote(cls, base_url: str, token: str = "") -> "AttendanceAPI":
        return cls(connect(base_url, token))

    def list_attendance(self, date_from: str | None = None,
                        date_to: str | None = None,
                        employee_id: str | None = None,
                        department: str | None = None) -> list:
        return self._call("employees", "list_attendance",
                          date_from, date_to, employee_id, department)

    def get_attendance(self, attendance_id: int) -> dict | None:
        return self._call("employees", "get_attendance", attendance_id)

    def pending_uploads(self, limit: int = 200) -> list:
        return self._call("employees", "pending_attendance", limit)

    def update_attendance(self, attendance_id: int, work_date: str,
                          check_in: str | None, check_out: str | None,
                          reason: str, status_override: str | None = None) -> dict:
        return self._call("attendance", "update_attendance", attendance_id,
                          work_date=work_date, check_in=check_in,
                          check_out=check_out, reason=reason,
                          status_override=status_override)

    def review_events(self) -> list:
        return self._call("operations", "review_events")

    def review(self, event_id: str, employee_id: str, reason: str) -> dict:
        return self._call("operations", "review", event_id, employee_id, reason)
