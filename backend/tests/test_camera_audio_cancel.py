from __future__ import annotations

import threading

from camera_tracking.audio.announcer import CameraCheckInAnnouncer


class _FakeTalkProcess:
    def __init__(self) -> None:
        self.killed = False

    def poll(self):
        return None

    def kill(self) -> None:
        self.killed = True


def test_cancel_playback_kills_visualtalk_child_and_clears_speaking() -> None:
    announcer = object.__new__(CameraCheckInAnnouncer)
    announcer._talk_process_lock = threading.Lock()
    announcer._talk_process = _FakeTalkProcess()
    announcer._speaking = threading.Event()
    announcer._speaking.set()

    announcer.cancel_playback()

    assert announcer._talk_process.killed is True
    assert announcer.speaking is False
