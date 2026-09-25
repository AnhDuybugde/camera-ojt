"""Server-side Supabase Auth and membership adapter.

Only short-lived app sessions leave this module. Supabase service credentials
and Auth bearer tokens are never returned to the Streamlit client.
"""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class SupabaseAuth:
    def __init__(self, url=None, anon_key=None, service_key=None, opener=urlopen):
        self.url = (url or os.getenv("SUPABASE_URL", "")).rstrip("/")
        self.anon_key = anon_key or os.getenv("SUPABASE_ANON_KEY", "")
        self.service_key = service_key or os.getenv("SUPABASE_SERVICE_KEY", "")
        self.opener = opener
        if not self.url.startswith("https://"):
            raise ValueError("Supabase Auth requires HTTPS")
        if not self.anon_key or not self.service_key:
            raise ValueError("Supabase Auth keys are not configured")

    def _request(self, path, payload=None, *, admin=False, access_token=None):
        key = self.service_key if admin else self.anon_key
        headers = {"apikey": key, "Content-Type": "application/json"}
        headers["Authorization"] = "Bearer " + (self.service_key if admin else
                                                  access_token or self.anon_key)
        if payload is not None and "on_conflict=" in path:
            headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
        request = Request(self.url + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers=headers, method="GET" if payload is None else "POST")
        try:
            with self.opener(request, timeout=12) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except HTTPError as error:
            # Do not propagate provider response text: it may contain PII.
            raise PermissionError("Supabase Auth rejected the request") from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("Supabase Auth is temporarily unavailable") from None

    def _membership(self, user_id):
        rows = self._request("/rest/v1/camera_memberships?user_id=eq." +
            quote(user_id, safe="") + "&select=user_id,employee_id,role,active",
            admin=True)
        if not isinstance(rows, list) or len(rows) != 1 or not rows[0].get("active", True):
            raise PermissionError("Account is not linked to an active employee")
        row = rows[0]
        role = str(row.get("role", "")).upper()
        if role not in {"ADMIN", "EMPLOYEE"}:
            raise PermissionError("Account role is invalid")
        employee_id = row.get("employee_id")
        if role == "EMPLOYEE" and not employee_id:
            raise PermissionError("Employee account has no employee mapping")
        return {"user_id": user_id, "employee_id": employee_id or "", "role": role}

    def login(self, email, password):
        if not isinstance(email, str) or "@" not in email or not password:
            raise ValueError("Enter your email and password")
        auth = self._request("/auth/v1/token?grant_type=password",
                             {"email": email.strip(), "password": password})
        return self._session(email, auth)

    def request_otp(self, email):
        if not isinstance(email, str) or "@" not in email:
            raise ValueError("Enter a valid email address")
        self._request("/auth/v1/otp", {"email": email.strip(), "create_user": False})
        return {"ok": True}

    def verify_otp(self, email, token, purpose="invite", new_password="", confirmation=""):
        if purpose not in {"email", "invite", "recovery"}:
            raise ValueError("Invalid email verification purpose")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Enter the code from your email")
        auth = self._request("/auth/v1/verify", {"email": email.strip(),
            "token": token.strip(), "type": purpose})
        access_token = auth.get("access_token")
        if new_password or confirmation:
            if new_password != confirmation or len(new_password) < 8:
                raise ValueError("New password must match and contain at least 8 characters")
            request = Request(self.url + "/auth/v1/user",
                data=json.dumps({"password": new_password}).encode(),
                headers={"apikey": self.anon_key, "Authorization": "Bearer " + str(access_token),
                         "Content-Type": "application/json"}, method="PUT")
            try:
                with self.opener(request, timeout=12) as response:
                    response.read()
            except (HTTPError, URLError, TimeoutError, OSError):
                raise RuntimeError("Could not set password") from None
        return self._session(email, auth)

    def _session(self, email, auth):
        user = auth.get("user") or {}
        user_id = user.get("id")
        if not user_id or not auth.get("access_token"):
            raise PermissionError("Email or password is incorrect")
        membership = self._membership(str(user_id))
        membership["expires_in"] = max(60, min(3600, int(auth.get("expires_in", 3600))))
        membership["email"] = str(user.get("email") or email).strip().lower()
        return membership

    def recover(self, email):
        if not isinstance(email, str) or "@" not in email:
            raise ValueError("Enter a valid email address")
        self._request("/auth/v1/recover", {"email": email.strip()})
        # Same response whether the address exists or not.
        return {"ok": True}

    def change_password(self, email, current, new, confirmation):
        if new != confirmation or len(new) < 8:
            raise ValueError("New password must match and contain at least 8 characters")
        auth = self._request("/auth/v1/token?grant_type=password",
                             {"email": email, "password": current})
        token = auth.get("access_token")
        if not token:
            raise PermissionError("Current password is incorrect")
        request = Request(self.url + "/auth/v1/user",
            data=json.dumps({"password": new}).encode(),
            headers={"apikey": self.anon_key, "Authorization": "Bearer " + token,
                     "Content-Type": "application/json"}, method="PUT")
        try:
            with self.opener(request, timeout=12) as response:
                response.read()
        except (HTTPError, URLError, TimeoutError, OSError):
            raise RuntimeError("Could not update password") from None
        return {"ok": True}

    def provision(self, email, role, employee_id=""):
        """Invite an Auth user then upsert its authorization mapping."""
        email = str(email).strip().lower()
        role = str(role).strip().lower()
        if "@" not in email or role not in {"admin", "employee"}:
            raise ValueError("Invalid invitation row")
        if role == "employee" and not employee_id:
            raise ValueError("Employee invitation requires employee_id")
        page = self._request("/auth/v1/admin/users?page=1&per_page=1000", admin=True)
        users = page.get("users", []) if isinstance(page, dict) else []
        existing = next((user for user in users
                         if str(user.get("email", "")).strip().lower() == email), None)
        if existing:
            user_id = existing.get("id")
            memberships = self._request("/rest/v1/camera_memberships?user_id=eq." +
                quote(str(user_id), safe="") + "&select=employee_id,role,active", admin=True)
            if memberships and (memberships[0].get("employee_id") != (employee_id or None)
                                or memberships[0].get("role") != role):
                raise ValueError("Email is already linked to a different account")
        else:
            invited = self._request("/auth/v1/invite", {"email": email}, admin=True)
            user_id = (invited.get("user") or invited).get("id")
        if not user_id:
            raise RuntimeError("Supabase did not return the invited user id")
        mapping = {"user_id": user_id, "employee_id": employee_id or None,
                   "role": role, "active": True}
        self._request("/rest/v1/camera_memberships?on_conflict=user_id", [mapping], admin=True)
        return {"invited": True, "role": role}
