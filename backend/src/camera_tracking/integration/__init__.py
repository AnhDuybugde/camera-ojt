"""Loose-coupled adapters used by the integrated team workspace."""

from camera_tracking.integration.be_xinh_bridge import (
    BackendStatusClient,
    BeXinhStatusBridge,
    PersonSnapshot,
    parse_people,
)

__all__ = [
    "BackendStatusClient",
    "BeXinhStatusBridge",
    "PersonSnapshot",
    "parse_people",
]
