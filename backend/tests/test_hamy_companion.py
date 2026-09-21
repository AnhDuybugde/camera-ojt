from __future__ import annotations

from camera_tracking.audio.companion import HamyCompanion, GROUP_ARRIVAL


class FakeSpeaker:
    enabled = True
    speaking = False

    def __init__(self):
        self.calls = []
        self.prewarmed = []

    def say(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return True

    def prewarm(self, texts, *, priority=90):
        texts = list(texts)
        self.prewarmed.extend(texts)
        return len(texts)

    def status(self):
        return {"ready": True, "speaking": False, "queue_depth": 0}


def test_single_arrival_is_delayed_only_for_group_window():
    speaker = FakeSpeaker()
    hamy = HamyCompanion(speaker, arrival_group_window_s=0.8)
    hamy.identify(global_id=1, person_id="p1", display_name="An", now_s=10.0)
    hamy.tick(10.5)
    assert not speaker.calls
    hamy.tick(10.81)
    assert len(speaker.calls) == 1
    assert "An" in speaker.calls[0][0]


def test_simultaneous_arrivals_merge_to_one_group_greeting():
    speaker = FakeSpeaker()
    hamy = HamyCompanion(speaker, arrival_group_window_s=0.8)
    hamy.identify(global_id=1, person_id="p1", display_name="An", now_s=10.0)
    hamy.identify(global_id=2, person_id="p2", display_name="Binh", now_s=10.2)
    hamy.tick(10.85)
    assert len(speaker.calls) == 1
    assert speaker.calls[0][0] == GROUP_ARRIVAL


def test_sitting_quiet_does_not_generate_random_banter():
    speaker = FakeSpeaker()
    hamy = HamyCompanion(
        speaker,
        arrival_group_window_s=0.2,
        water_after_s=2700,
        rest_after_s=5400,
    )
    hamy.identify(global_id=1, person_id="p1", display_name="An", now_s=0.0)
    hamy.tick(0.3)
    speaker.calls.clear()
    for now_s in (30.0, 300.0, 1200.0, 2400.0):
        hamy.observe_presence(
            person_id="p1",
            display_name="An",
            now_s=now_s,
            stationary_since_s=0.0,
            global_id=1,
        )
        hamy.tick(now_s)
    assert speaker.calls == []


def test_wave_has_high_priority_and_works_without_identity():
    speaker = FakeSpeaker()
    hamy = HamyCompanion(speaker)
    assert hamy.wave(now_s=10.0, global_id=99)
    assert speaker.calls[-1][1]["priority"] == 0
