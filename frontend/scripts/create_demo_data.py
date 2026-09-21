"""Insert demo employees without fake face embeddings."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.db import Database  # noqa: E402


DEMO_EMPLOYEES = [
    {"employee_id": "NV001", "full_name": "Nguyen Van A", "department": "AI Department", "position": "AI Engineer"},
    {"employee_id": "NV002", "full_name": "Tran Thi B", "department": "Human Resources", "position": "HR Specialist"},
    {"employee_id": "NV003", "full_name": "Le Van C", "department": "Operations", "position": "Coordinator"},
]


def main() -> None:
    db = Database()
    added = 0
    for employee in DEMO_EMPLOYEES:
        try:
            db.add_employee(employee)
            added += 1
        except sqlite3.IntegrityError:
            print(f"Skipped existing employee: {employee['employee_id']}")
    print(f"Added {added} demo employee(s).")


if __name__ == "__main__":
    main()

