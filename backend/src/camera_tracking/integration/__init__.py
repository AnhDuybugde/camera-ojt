"""Loose-coupled adapters used by the integrated team workspace."""

from camera_tracking.integration.be_xinh_bridge import (
    BackendStatusClient,
    BeXinhStatusBridge,
    PersonSnapshot,
    audio_event_stream_present,
    parse_audio_events,
    parse_people,
)

__all__ = [
    "BackendStatusClient",
    "BeXinhStatusBridge",
    "PersonSnapshot",
    "audio_event_stream_present",
    "parse_audio_events",
    "parse_people",
]
