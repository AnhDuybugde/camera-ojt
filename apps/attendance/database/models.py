"""Compatibility shim — canonical home is backend.app.database.models."""
from backend.app.database.models import Attendance, Employee

__all__ = ["Attendance", "Employee"]
