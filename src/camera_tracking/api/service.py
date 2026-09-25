"""Compatibility shim — canonical home is backend.app.api.service."""
from backend.app.api.service import (
    AUTH_METHODS,
    DB_METHODS,
    EMPLOYEE_METHODS,
    ApplicationAPI,
)

__all__ = ["AUTH_METHODS", "DB_METHODS", "EMPLOYEE_METHODS", "ApplicationAPI"]
