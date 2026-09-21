"""Small pure helper functions."""
from __future__ import annotations

import re
from datetime import date, datetime


EMPLOYEE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{2,32}$")


def validate_employee_id(value: str) -> str:
    value = value.strip().upper()
    if not EMPLOYEE_ID_PATTERN.fullmatch(value):
        raise ValueError("Employee ID must contain 2-32 letters, numbers, '_' or '-'.")
    return value


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def iso_today() -> str:
    return date.today().isoformat()

