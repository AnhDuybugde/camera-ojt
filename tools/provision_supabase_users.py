"""Dry-run-first invitation/mapping import; source CSV must remain private."""
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "apps" / "attendance"))


def read_rows(path, employees):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["employee_id", "email", "role"]:
            raise ValueError("CSV header must be employee_id,email,role")
        rows, ids, emails = [], set(), set()
        for line, row in enumerate(reader, 2):
            employee_id = (row.get("employee_id") or "").strip()
            email = (row.get("email") or "").strip().lower()
            role = (row.get("role") or "").strip().lower()
            if "@" not in email or role not in {"admin", "employee"}:
                raise ValueError(f"Invalid account row at line {line}")
            if email in emails:
                raise ValueError(f"Duplicate email at line {line}")
            emails.add(email)
            if role == "employee":
                if not employee_id or employee_id not in employees or employee_id in ids:
                    raise ValueError(f"Invalid/duplicate employee_id at line {line}")
                ids.add(employee_id)
            elif employee_id:
                raise ValueError(f"Admin employee_id must be blank at line {line}")
            rows.append({"employee_id": employee_id, "email": email, "role": role})
    if not rows or not any(row["role"] == "admin" for row in rows):
        raise ValueError("CSV must include at least one admin account")
    if ids != employees:
        raise ValueError("CSV must map every current employee exactly once")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ROOT / "apps/attendance/data/supabase-users.csv"))
    parser.add_argument("--apply", action="store_true", help="send invitations and provision memberships")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from camera_tracking.application.backend import load_services
    from backend.app.api.supabase_auth import SupabaseAuth
    db, *_ = load_services()
    employee_ids = {row["employee_id"] for row in db.list_employees()}
    rows = read_rows(args.csv, employee_ids)
    print(f"Validated {len(rows)} account rows; apply={args.apply}.")
    if args.apply:
        auth = SupabaseAuth()
        sent = 0
        for row in rows:
            auth.provision(row["email"], row["role"], row["employee_id"])
            sent += 1
        print(f"Accounts linked: {sent}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("Account import failed:", type(error).__name__)
        raise SystemExit(1)
