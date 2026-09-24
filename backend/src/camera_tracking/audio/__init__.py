"""Camera AIM audio + event-driven Bé Xinh companion."""

from camera_tracking.audio.announcer import CameraCheckInAnnouncer
from camera_tracking.audio.companion import HamyCompanion
from camera_tracking.audio.event_router import AudioEventRouter, RouteResult
from camera_tracking.audio.events import (
    AudioEvent,
    AudioEventKind,
    audio_event_stream_present,
    parse_audio_events,
)
from camera_tracking.audio.gesture import GestureEvent, HandGestureDetector
from camera_tracking.audio.interaction import InteractionEngine, InteractionEvent, MotionSnapshot
from camera_tracking.audio.policy import AudioPolicy, EventPolicy
from camera_tracking.audio.zone_observer import ImageZone, ZoneEventObserver

__all__ = [
    "AudioEvent",
    "AudioEventKind",
    "AudioEventRouter",
    "AudioPolicy",
    "CameraCheckInAnnouncer",
    "EventPolicy",
    "HamyCompanion",
    "GestureEvent",
    "HandGestureDetector",
    "InteractionEngine",
    "InteractionEvent",
    "ImageZone",
    "MotionSnapshot",
    "RouteResult",
    "ZoneEventObserver",
    "audio_event_stream_present",
    "parse_audio_events",
]
