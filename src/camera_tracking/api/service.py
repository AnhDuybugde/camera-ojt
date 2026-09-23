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
AUTH_METHODS = frozenset({"change_password", "recover", "provision"})
EMPLOYEE_METHODS = frozenset({"get_employee", "list_employees", "list_attendance",
    "get_work_schedule", "list_work_schedules", "save_work_schedule", "save_work_schedules",
    "save_embedding", "save_embedding_samples", "detect"})


class ApplicationAPI:
    def __init__(self, db, auth, attendance_admin, sync, operations=None, enrollment=None):
        self.db, self.auth = db, auth
        self.services = {"employees": db, "auth": auth, "attendance": attendance_admin,
                         "sync": sync, "operations": operations, "enrollment": enrollment}
        self.sessions = {}
        self.lock = threading.Lock()

    def login(self, email, password):
        account = self.auth.login(email, password)
        return self._start_session(account)

    def _start_session(self, account):
        role, employee_id = account["role"], account["employee_id"]
        token = secrets.token_urlsafe(32)
        with self.lock:
            now = time.monotonic()
            self.sessions = {k: v for k, v in self.sessions.items() if v[4] > now}
            self.sessions[token] = (role, employee_id, account["user_id"],
                                    account["email"], now + account["expires_in"])
        return {"token": token, "role": role, "employee_id": employee_id,
                "email": account["email"], "expires_in": account["expires_in"]}

    def dispatch(self, service, method, args, kwargs, token):
        if (service, method) == ("auth", "login"):
            return self.login(*args, **kwargs)
        if (service, method) == ("auth", "recover"):
            return self.auth.recover(*args, **kwargs)
        if (service, method) == ("auth", "request_otp"):
            return self.auth.request_otp(*args, **kwargs)
        if (service, method) == ("auth", "verify_otp"):
            return self._start_session(self.auth.verify_otp(*args, **kwargs))
        with self.lock:
            session = self.sessions.get(token)
        if session is None or session[4] <= time.monotonic():
            raise PermissionError("Session expired. Please sign in again.")
        role, employee_id, user_id, email, _ = session
        allowed = {"employees": DB_METHODS, "auth": AUTH_METHODS,
                   "attendance": {"update_attendance"}, "sync": {"sync_pending"},
                   "operations": {"diagnostics", "review_events", "review", "conflicts", "resolve_conflict"},
                   "enrollment": {"detect"}}
        if method not in allowed.get(service, set()):
            raise PermissionError("Operation is not exposed")
        if role == "EMPLOYEE":
            if method not in EMPLOYEE_METHODS:
                raise PermissionError("Administrator access required")
        if method == "provision" and role != "ADMIN":
            raise PermissionError("Administrator access required")
        if method == "change_password":
            if len(args) == 3:
                kwargs = {**kwargs, "email": email, "current": args[0],
                          "new": args[1], "confirmation": args[2]}
            args = ()
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
        return result
