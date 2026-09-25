"""Compatibility shim — canonical home is backend.app.services.room_status_service."""
from backend.app.services.room_status_service import (
    LABEL_AT_DOOR,
    LABEL_AWAY,
    LABEL_OUT_OF_DOOR,
    LABEL_UNKNOWN,
    LABEL_WORKING,
    RoomPersonStatus,
    RoomPresenceAggregator,
)

__all__ = [
    "LABEL_AT_DOOR",
    "LABEL_AWAY",
    "LABEL_OUT_OF_DOOR",
    "LABEL_UNKNOWN",
    "LABEL_WORKING",
    "RoomPersonStatus",
    "RoomPresenceAggregator",
]
