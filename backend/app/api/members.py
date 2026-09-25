"""Members domain router — employee profiles + work schedules.

Typed wrapper over the ``employees`` API service (same calls the Streamlit
frontend makes through ``RemoteService``). Transport-agnostic: pass any
``call(service, method, *args, **kwargs)`` — see :func:`connect`.
"""
from __future__ import annotations

from collections.abc import Callable

from backend.app.api._http import connect


class MembersAPI:
    def __init__(self, call: Callable):
        self._call = call

    @classmethod
    def remote(cls, base_url: str, token: str = "") -> "MembersAPI":
        return cls(connect(base_url, token))

    def list_members(self) -> list:
        return self._call("employees", "list_employees")

    def get_member(self, employee_id: str) -> dict | None:
        return self._call("employees", "get_employee", employee_id)

    def add_member(self, data: dict) -> None:
        return self._call("employees", "add_employee", data)

    def update_member(self, employee_id: str, data: dict) -> None:
        return self._call("employees", "update_employee", employee_id, data)

    def delete_member(self, employee_id: str) -> None:
        return self._call("employees", "delete_employee", employee_id)

    def get_schedule(self, employee_id: str, work_date: str,
                     work_session: str = "MORNING") -> dict | None:
        return self._call("employees", "get_work_schedule",
                          employee_id, work_date, work_session)

    def list_schedules(self, date_from: str, date_to: str,
                       employee_id: str | None = None,
                       department: str | None = None) -> list:
        return self._call("employees", "list_work_schedules",
                          date_from, date_to, employee_id, department)

    def save_schedule(self, employee_id: str, work_date: str,
                      work_status: str, work_session: str = "MORNING") -> None:
        return self._call("employees", "save_work_schedule",
                          employee_id, work_date, work_status, work_session)

    def save_schedules(self, entries: list) -> None:
        return self._call("employees", "save_work_schedules", entries)

    def audit_logs(self, limit: int = 500) -> list:
        return self._call("employees", "list_audit_logs", limit)
