"""Domain models independent from the persistence layer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Employee:
    employee_id: str
    full_name: str
    department: str = ""
    position: str = ""
    email: str = ""
    phone: str = ""
    created_date: str = ""
    has_face: bool = False


@dataclass(slots=True)
class Attendance:
    id: int
    employee_id: str
    employee_name: str
    department: str
    date: str
    check_in: str | None
    check_out: str | None
    status: str
    sync_status: str
    created_at: str
    updated_at: str

