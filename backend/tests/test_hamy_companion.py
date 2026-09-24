from __future__ import annotations

from camera_tracking.audio.companion import HamyCompanion


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


class SelectiveCacheSpeaker(FakeSpeaker):
    def __init__(self, cached):
        super().__init__()
        self.cached = set(cached)

    def is_cached(self, text):
        return text in self.cached


def test_named_wave_prefers_cached_line_for_immediate_response():
    cached = "Hihi, An chào Bé Xinh hả? Chào nha."
    speaker = SelectiveCacheSpeaker({cached})
    hamy = HamyCompanion(speaker)

    assert hamy.wave(
        now_s=10.0, person_id="p1", display_name="An", global_id=1
    )
    assert speaker.calls[0][0] == cached


def test_single_arrival_waits_for_stable_presence_delay():
    speaker = FakeSpeaker()
    hamy = HamyCompanion(
        speaker, arrival_delay_s=5.0, arrival_group_window_s=0.8
    )
    hamy.identify(global_id=1, person_id="p1", display_name="An", now_s=10.0)
    hamy.observe_presence(
        person_id="p1", display_name="An", now_s=15.0, global_id=1
    )
    hamy.tick(15.5)
    assert not speaker.calls
    hamy.observe_presence(
        person_id="p1", display_name="An", now_s=15.81, global_id=1
    )
    hamy.tick(15.81)
    assert len(speaker.calls) == 1
    assert "An" in speaker.calls[0][0]


def test_simultaneous_arrivals_merge_to_one_group_greeting():
    speaker = FakeSpeaker()
    hamy = HamyCompanion(
        speaker, arrival_delay_s=5.0, arrival_group_window_s=0.8
    )
    hamy.identify(global_id=1, person_id="p1", display_name="An", now_s=10.0)
    hamy.identify(global_id=2, person_id="p2", display_name="Binh", now_s=10.2)
    hamy.observe_presence(
        person_id="p1", display_name="An", now_s=15.85, global_id=1
    )
    hamy.observe_presence(
        person_id="p2", display_name="Binh", now_s=15.85, global_id=2
    )
    hamy.tick(15.85)
    assert len(speaker.calls) == 1
    assert "An" in speaker.calls[0][0]
    assert "Binh" in speaker.calls[0][0] or "mọi người" in speaker.calls[0][0]


def test_sitting_quiet_generates_banter_only_after_threshold():
    speaker = FakeSpeaker()
    hamy = HamyCompanion(
        speaker,
        arrival_delay_s=0.0,
        arrival_group_window_s=0.2,
        water_after_s=2700,
        rest_after_s=5400,
        banter_after_s=1200,
        banter_repeat_s=1800,
    )
    hamy.identify(global_id=1, person_id="p1", display_name="An", now_s=0.0)
    hamy.tick(0.3)
    speaker.calls.clear()
    for now_s in (30.0, 300.0, 1199.0):
        hamy.observe_presence(
            person_id="p1",
            display_name="An",
            now_s=now_s,
            stationary_since_s=0.0,
            global_id=1,
        )
        hamy.tick(now_s)
    assert speaker.calls == []

    hamy.observe_presence(
        person_id="p1",
        display_name="An",
        now_s=1201.0,
        stationary_since_s=0.0,
        global_id=1,
    )
    hamy.tick(1201.0)
    assert len(speaker.calls) == 1
    assert "Bé Xinh" in speaker.calls[0][0]


def test_wave_has_high_priority_and_works_without_identity():
    speaker = FakeSpeaker()
    hamy = HamyCompanion(speaker)
    assert hamy.wave(now_s=10.0, global_id=99)
    assert speaker.calls[-1][1]["priority"] == 0


def test_zone_mode_disables_presence_arrival_but_keeps_wave() -> None:
    speaker = FakeSpeaker()
    hamy = HamyCompanion(
        speaker, arrival_delay_s=0.0, arrival_group_window_s=0.2
    )
    hamy.set_auto_arrival_from_presence(False)
    hamy.identify(global_id=1, person_id="p1", display_name="An", now_s=0.0)
    hamy.observe_presence(
        person_id="p1", display_name="An", now_s=1.0, global_id=1
    )
    hamy.tick(1.0)
    assert speaker.calls == []
    assert hamy.status()["mode"] == "zone_events"
    assert hamy.wave(now_s=2.0, person_id="p1", display_name="An", global_id=1)


def test_wave_rotates_friendly_lines_and_postpones_banter():
    speaker = FakeSpeaker()
    hamy = HamyCompanion(
        speaker,
        arrival_delay_s=0.0,
        arrival_group_window_s=0.2,
        banter_after_s=300,
        banter_repeat_s=600,
    )
    hamy.identify(global_id=1, person_id="p1", display_name="An", now_s=0.0)
    hamy.tick(0.3)
    speaker.calls.clear()

    assert hamy.wave(now_s=10.0, person_id="p1", display_name="An", global_id=1)
    assert hamy.wave(now_s=26.0, person_id="p1", display_name="An", global_id=1)
    assert speaker.calls[0][0] != speaker.calls[1][0]
    assert all("An" in call[0] for call in speaker.calls)

    speaker.calls.clear()
    hamy.observe_presence(
        person_id="p1",
        display_name="An",
        now_s=325.0,
        stationary_since_s=0.0,
        global_id=1,
    )
    hamy.tick(325.0)
    assert speaker.calls == []
    hamy.observe_presence(
        person_id="p1",
        display_name="An",
        now_s=327.0,
        stationary_since_s=0.0,
        global_id=1,
    )
    hamy.tick(327.0)
    assert len(speaker.calls) == 1


def test_wave_cancels_pending_delayed_arrival():
    speaker = FakeSpeaker()
    hamy = HamyCompanion(
        speaker, arrival_delay_s=5.0, arrival_group_window_s=0.2
    )
    hamy.identify(global_id=1, person_id="p1", display_name="An", now_s=0.0)
    assert hamy.wave(
        now_s=1.0, person_id="p1", display_name="An", global_id=1
    )
    hamy.observe_presence(
        person_id="p1", display_name="An", now_s=6.0, global_id=1
    )
    hamy.tick(6.0)
    assert len(speaker.calls) == 1


def test_wave_has_global_gap_even_when_gid_changes():
    speaker = FakeSpeaker()
    hamy = HamyCompanion(speaker, wave_global_gap_s=15.0)
    assert hamy.wave(now_s=1.0, global_id=10)
    assert not hamy.wave(now_s=8.0, global_id=11)
    assert hamy.wave(now_s=16.1, global_id=11)


def test_near_mode_waits_for_proximity_then_greets_with_name():
    speaker = FakeSpeaker()
    hamy = HamyCompanion(
        speaker,
        arrival_delay_s=5.0,
        arrival_group_window_s=0.1,
        greet_only_when_near=True,
    )
    hamy.identify(global_id=1, person_id="p1", display_name="An", now_s=0.0)
    hamy.observe_presence(
        person_id="p1", display_name="An", now_s=6.0, global_id=1,
        near_camera=False, estimated_distance_m=1.2,
    )
    hamy.tick(6.0)
    assert speaker.calls == []

    hamy.observe_presence(
        person_id="p1", display_name="An", now_s=6.2, global_id=1,
        near_camera=True, estimated_distance_m=0.5,
    )
    hamy.tick(6.2)
    assert len(speaker.calls) == 1
    assert "An" in speaker.calls[0][0]
    assert speaker.calls[0][1]["priority"] == 10
