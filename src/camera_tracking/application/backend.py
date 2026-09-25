"""Compatibility shim — canonical home is backend.app.main."""
from backend.app.main import load_services, run_backend

__all__ = ["load_services", "run_backend"]
