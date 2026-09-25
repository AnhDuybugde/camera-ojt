"""Compatibility shim — canonical home is frontend.src.services.backend."""
from frontend.src.services.backend import (
    BackendWorker,
    RemoteDetector,
    RemoteRecognizer,
    RemoteService,
)

__all__ = ["BackendWorker", "RemoteDetector", "RemoteRecognizer", "RemoteService"]
