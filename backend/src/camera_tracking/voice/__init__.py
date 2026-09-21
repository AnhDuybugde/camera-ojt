"""Speech generation and camera talk outputs (legacy WebSDK + P2P VisualTalk)."""

from camera_tracking.voice.greeter import VoiceGreeter
from camera_tracking.voice.voice_trigger import (
    DEFAULT_TRIGGER_WORDS,
    VoiceTrigger,
    is_voice_trigger,
    normalize_trigger_text,
)
from camera_tracking.voice.imou_bridge import (
    AudioTalkError,
    ImouAudioTalkBridge,
    ImouAudioTalkOutput,
    ImouCredentials,
)
from camera_tracking.voice.p2p_talk import (
    ImouP2PCredentials,
    ImouP2PTalkOutput,
    P2PTalkError,
)

__all__ = [
    "AudioTalkError",
    "ImouAudioTalkBridge",
    "ImouAudioTalkOutput",
    "ImouCredentials",
    "ImouP2PCredentials",
    "ImouP2PTalkOutput",
    "P2PTalkError",
    "VoiceGreeter",
    "DEFAULT_TRIGGER_WORDS",
    "VoiceTrigger",
    "is_voice_trigger",
    "normalize_trigger_text",
]
