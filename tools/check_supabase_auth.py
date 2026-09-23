"""Create, authenticate and remove one synthetic Supabase Auth account."""
from pathlib import Path
import json
import os
import secrets
import sys
import uuid
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def send(url, key, path, payload=None, method=None):
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(url + path, data=body, method=method or ("POST" if body else "GET"),
        headers={"apikey": key, "Authorization": "Bearer " + key,
                 "Content-Type": "application/json",
                 "Prefer": "resolution=merge-duplicates,return=minimal"})
    with urlopen(request, timeout=12) as response:
        raw = response.read()
        return json.loads(raw) if raw else {}


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from camera_tracking.api.supabase_auth import SupabaseAuth

    base = os.environ["SUPABASE_URL"].rstrip("/")
    service = os.environ["SUPABASE_SERVICE_KEY"]
    email = "codex-probe+" + uuid.uuid4().hex + "@example.invalid"
    password = secrets.token_urlsafe(28)
    user_id = None
    try:
        created = send(base, service, "/auth/v1/admin/users",
                       {"email": email, "password": password, "email_confirm": True})
        user_id = (created.get("user") or created).get("id")
        if not user_id:
            raise RuntimeError("Synthetic Auth user was not created")
        send(base, service, "/rest/v1/camera_memberships?on_conflict=user_id", [{
            "user_id": user_id, "employee_id": "synthetic-probe", "role": "employee", "active": True
        }])
        result = SupabaseAuth().login(email, password)
        if result["user_id"] != user_id or result["employee_id"] != "synthetic-probe" \
                or result["role"] != "EMPLOYEE":
            raise RuntimeError("Auth-to-membership mapping did not match")
    finally:
        if user_id:
            send(base, service, "/auth/v1/admin/users/" + user_id, method="DELETE")
    print({"provider": "supabase_auth", "status": "ok", "mapping": "verified",
           "cleanup": "complete"})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Avoid printing provider bodies, credentials or the generated account.
        print("Supabase Auth probe failed:", type(error).__name__)
        raise SystemExit(1)
