"""Vendored P2P VisualTalk sender (nguon: test-sound-camera-imou).

Goc la implementation Python-stdlib VisualTalk sender dan xuat tu public
research project home-assistant-tools/imou-life. Chi giu lai duong code toi
thieu de gui WAV -> loa camera Imou qua DHP2P/PTCP relay tunnel (remote
port 8086), khong dung Chrome, virtual mic, OpenSDK browser hay record mic
laptop.

Xem docs/P2P_TALK.md de biet cach cau hinh + giay phep upstream.
"""

from camera_tracking.voice.p2p.imou_dhav import build_frames, pack_dhav_audio
from camera_tracking.voice.p2p.imou_dhp2p import DHP2PTunnel, p2p_handshake
from camera_tracking.voice.p2p.imou_visualtalk import VisualTalkClient
from camera_tracking.voice.p2p.imou_wsse import password_digest

__all__ = [
    "DHP2PTunnel",
    "VisualTalkClient",
    "build_frames",
    "p2p_handshake",
    "pack_dhav_audio",
    "password_digest",
]
