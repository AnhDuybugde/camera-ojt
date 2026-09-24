"""Preview or explicitly backfill past ON days that have no attendance row.

No dates are inferred: --from and --to are required. Preview is the default;
--apply inserts only missing employee/day rows and writes SYSTEM audit events.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.db import Database  # noqa: E402


def candidates(db: Database, first: date, last: date) -> list[tuple[str, str]]:
    schedules = db.list_work_schedules(first.isoformat(), last.isoformat())
    tracked = {(row["employee_id"], row["work_date"])
               for row in schedules if row["work_status"] == "ON"}
    existing = {(row["employee_id"], row["date"])
                for row in db.list_attendance(first.isoformat(), last.isoformat())}
    return sorted(tracked - existing)


def backfill(db: Database, first: date, last: date, *, apply: bool = False) -> tuple[int, int]:
    if first > last or last >= date.today():
        raise ValueError("Choose a valid past date range; current/future days use the live service")
    missing = candidates(db, first, last)
    if not apply:
        return len(missing), 0
    employees = {row["employee_id"]: row for row in db.list_employees()}
    timestamp = datetime.now().isoformat(timespec="seconds")
    inserted = 0
    with db.transaction() as conn:
        for employee_id, work_date in missing:
            employee = employees.get(employee_id)
            if employee is None:
                continue
            result = conn.execute(
                """INSERT INTO attendance
                   (employee_id,employee_name,department,date,check_in,check_out,status,
                    presence_status,sync_status,created_at,updated_at)
                   VALUES (?,?,?,?,NULL,NULL,'ABSENT','ABSENT','PENDING',?,?)
                   ON CONFLICT(employee_id,date) DO NOTHING""",
                (employee_id, employee["full_name"], employee["department"],
                 work_date, timestamp, timestamp),
            )
            if result.rowcount:
                conn.execute(
                    """INSERT INTO audit_logs(timestamp,role,action,employee_id,old_values,new_values,reason)
                       VALUES (?,'SYSTEM','BACKFILL_ABSENT',?,?,?,?)""",
                    (timestamp, employee_id, "{}",
                     json.dumps({"date": work_date, "status": "ABSENT"}),
                     "Backfill có kiểm soát cho ngày ON không có bản ghi điểm danh"),
                )
                inserted += 1
    return len(missing), inserted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="first", type=date.fromisoformat, required=True)
    parser.add_argument("--to", dest="last", type=date.fromisoformat, required=True)
    parser.add_argument("--apply", action="store_true", help="Write missing rows; omit to preview only")
    args = parser.parse_args()
    found, inserted = backfill(Database(), args.first, args.last, apply=args.apply)
    print(f"Missing ON employee-days: {found}; inserted: {inserted}; mode: {'APPLY' if args.apply else 'PREVIEW'}")


if __name__ == "__main__":
    main()
