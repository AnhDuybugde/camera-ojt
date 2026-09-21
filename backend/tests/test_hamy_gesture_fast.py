from __future__ import annotations

from types import SimpleNamespace

from camera_tracking.audio.companion import GENERIC_WAVE, HamyCompanion
from camera_tracking.audio.gesture import HandGestureDetector


class _Speaker:
    enabled = True
    speaking = False

    def __init__(self):
        self.calls = []

    def say(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return True

    def prewarm(self, texts, *, priority=90):
        return 0

    def status(self):
        return {"ready": True, "speaking": False, "queue_depth": 0}


def test_wave_uses_wave_fallback():
    speaker = _Speaker()
    companion = HamyCompanion(speaker)
    assert companion.wave(now_s=1.0, global_id=7)
    _text, kwargs = speaker.calls[-1]
    assert kwargs["fallback_text"] == GENERIC_WAVE


def test_open_palm_four_long_fingers_scores_above_default_threshold():
    # Synthetic normalised hand: wrist at bottom, four fingertips far above
    # their PIP joints, thumb also extended, and index/pinky spread.
    pts = [SimpleNamespace(x=0.5, y=0.8) for _ in range(21)]
    pts[0] = SimpleNamespace(x=0.5, y=0.90)
    pts[3] = SimpleNamespace(x=0.40, y=0.66)
    pts[4] = SimpleNamespace(x=0.25, y=0.55)
    for tip, pip, x in ((8,6,0.25),(12,10,0.42),(16,14,0.60),(20,18,0.78)):
        pts[pip] = SimpleNamespace(x=x, y=0.58)
        pts[tip] = SimpleNamespace(x=x, y=0.22)
    score = HandGestureDetector._open_palm_confidence(pts)
    assert score >= 0.68
