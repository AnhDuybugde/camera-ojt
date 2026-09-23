"""Authenticated facade; no SQL or arbitrary Python method dispatch is exposed."""
from __future__ import annotations

import inspect
import secrets
import threading
import time


DB_METHODS = frozenset({
    "get_employee", "list_employees", "add_employee", "update_employee", "delete_employee",
    "save_embedding", "list_attendance", "get_attendance", "list_audit_logs",
    "get_work_schedule", "list_work_schedules", "save_work_schedule", "save_work_schedules",
    "pending_attendance", "save_embedding_samples",
})
AUTH_METHODS = frozenset({"account", "ensure_accounts", "change_employee_password",
                          "reset_employee_password", "change_password"})
EMPLOYEE_METHODS = frozenset({"get_employee", "list_employees", "list_attendance",
    "get_work_schedule", "list_work_schedules", "save_work_schedule", "save_work_schedules",
    "save_embedding", "save_embedding_samples", "account", "change_employee_password", "detect"})


class ApplicationAPI:
    def __init__(self, db, auth, attendance_admin, sync, operations=None, enrollment=None):
        self.db, self.auth = db, auth
        self.services = {"employees": db, "auth": auth, "attendance": attendance_admin,
                         "sync": sync, "operations": operations, "enrollment": enrollment}
        self.sessions = {}
        self.lock = threading.Lock()

    def _version(self, role, employee_id):
        if role == "EMPLOYEE":
            account = self.auth.account(employee_id)
            return account.get("version") if account else None
        row = self.db.fetch_one("SELECT password_hash FROM auth_settings WHERE role='ADMIN'")
        return row["password_hash"] if row else None

    def login(self, role, password, employee_id=""):
        if role not in {"ADMIN", "EMPLOYEE"}:
            raise ValueError("Invalid role")
        account = self.auth.login(role, password, employee_id)
        if account is None:
            return None
        token = secrets.token_urlsafe(32)
        with self.lock:
            now = time.monotonic()
            self.sessions = {k: v for k, v in self.sessions.items() if v[3] > now}
            self.sessions[token] = (role, employee_id, self._version(role, employee_id), now + 28800)
        return {"token": token, "version": account.get("version"),
                "must_change": account.get("must_change", False)}

    def dispatch(self, service, method, args, kwargs, token):
        if (service, method) == ("auth", "login"):
            return self.login(*args, **kwargs)
        with self.lock:
            session = self.sessions.get(token)
        if session is None or session[3] <= time.monotonic():
            raise PermissionError("Session expired. Please sign in again.")
        role, employee_id, version, _ = session
        if self._version(role, employee_id) != version:
            raise PermissionError("Account changed. Please sign in again.")
        allowed = {"employees": DB_METHODS, "auth": AUTH_METHODS,
                   "attendance": {"update_attendance"}, "sync": {"sync_pending"},
                   "operations": {"diagnostics", "review_events", "review", "conflicts", "resolve_conflict"},
                   "enrollment": {"detect"}}
        if method not in allowed.get(service, set()):
            raise PermissionError("Operation is not exposed")
        if role == "EMPLOYEE":
            account = self.auth.account(employee_id)
            if account["must_change"] and method not in {"account", "change_employee_password"}:
                raise PermissionError("Change your initial password first")
            if method not in EMPLOYEE_METHODS:
                raise PermissionError("Administrator access required")
        fn = getattr(self.services[service], method)
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        if "actor_role" in inspect.signature(fn).parameters:
            bound.arguments["actor_role"] = role
        if "actor_employee_id" in inspect.signature(fn).parameters:
            bound.arguments["actor_employee_id"] = employee_id or None
        if role == "EMPLOYEE":
            if "employee_id" in inspect.signature(fn).parameters:
                requested = bound.arguments.get("employee_id")
                if method == "list_attendance":
                    bound.arguments["employee_id"] = employee_id
                elif requested and requested != employee_id:
                    raise PermissionError("Only your own records are accessible")
                else:
                    bound.arguments["employee_id"] = employee_id
            if method in {"save_embedding", "save_embedding_samples"} and self.db.get_employee(employee_id)["has_face"]:
                raise PermissionError("Only an administrator can replace enrollment")
        result = fn(*bound.args, **bound.kwargs)
        if method == "list_employees" and role == "EMPLOYEE":
            result = [row for row in result if row["employee_id"] == employee_id]
        if method == "account" and result:
            result = {key: result[key] for key in ("employee_id", "version", "must_change")}
        if method in {"change_employee_password", "change_password"}:
            with self.lock:
                self.sessions[token] = (role, employee_id, self._version(role, employee_id), session[3])
        return result
