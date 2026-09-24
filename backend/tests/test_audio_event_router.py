from __future__ import annotations

from dataclasses import replace

from camera_tracking.audio.event_router import AudioEventRouter
from camera_tracking.audio.events import AudioEvent, AudioEventKind
from camera_tracking.audio.policy import AudioPolicy
from camera_tracking.audio.phrases import iter_prewarm_texts


class FakeSpeaker:
    enabled = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def say(self, text: str, **kwargs) -> bool:
        self.calls.append((text, kwargs))
        return True


def event(
    event_id: str,
    kind: AudioEventKind,
    *,
    source_type: str | None = None,
    person_id: str = "NV01",
    name: str = "Minh",
    gid: int | None = 7,
    timestamp: float = 1.0,
    dwell: float | None = None,
) -> AudioEvent:
    return AudioEvent(
        event_id=event_id,
        kind=kind,
        timestamp=timestamp,
        source_type=source_type or kind.value,
        global_id=gid,
        person_id=person_id,
        display_name=name,
        dwell_seconds=dwell,
    )


def quiet_policy() -> AudioPolicy:
    return replace(AudioPolicy.from_env(), global_gap_s=0.0, same_person_gap_s=0.0)


def test_restroom_never_leaks_name_to_text_or_fallback() -> None:
    speaker = FakeSpeaker()
    router = AudioEventRouter(speaker, quiet_policy())
    result = router.route(event("wc-1", AudioEventKind.RESTROOM), now_s=10.0)
    assert result.accepted
    text, options = speaker.calls[-1]
    assert "Minh" not in text
    assert "Minh" not in options["fallback_text"]
    assert "Minh" not in router.status()["last_line"]


def test_duplicate_event_is_spoken_once() -> None:
    speaker = FakeSpeaker()
    router = AudioEventRouter(speaker, quiet_policy())
    item = event("door-1", AudioEventKind.DOOR_ENTER)
    assert router.route(item, now_s=10.0).accepted
    assert router.route(item, now_s=20.0).reason == "duplicate"
    assert len(speaker.calls) == 1


def test_per_person_event_cooldown_blocks_tracking_jitter() -> None:
    router = AudioEventRouter(FakeSpeaker(), quiet_policy())
    assert router.route(event("door-1", AudioEventKind.DOOR_ENTER), now_s=10.0).accepted
    assert router.route(event("door-2", AudioEventKind.DOOR_ENTER), now_s=20.0).reason == "cooldown"


def test_same_person_gap_suppresses_lower_priority_followup() -> None:
    policy = replace(AudioPolicy.from_env(), global_gap_s=0.0, same_person_gap_s=8.0)
    router = AudioEventRouter(FakeSpeaker(), policy)
    assert router.route(event("in", AudioEventKind.DOOR_ENTER), now_s=10.0).accepted
    assert router.route(event("near", AudioEventKind.BE_XINH_NEAR), now_s=12.0).reason == "person_gap"


def test_higher_priority_wave_can_pass_global_gap() -> None:
    policy = replace(AudioPolicy.from_env(), global_gap_s=10.0, same_person_gap_s=0.0)
    router = AudioEventRouter(FakeSpeaker(), policy)
    assert router.route(event("water", AudioEventKind.WATER), now_s=10.0).accepted
    assert router.route(event("wave", AudioEventKind.WAVE), now_s=11.0).accepted


def test_generic_dwell_can_mature_with_stable_event_id() -> None:
    speaker = FakeSpeaker()
    router = AudioEventRouter(speaker, quiet_policy())
    short = event("water-dwell", AudioEventKind.WATER, source_type="zone_dwell", dwell=0.5)
    enough = event("water-dwell", AudioEventKind.WATER, source_type="zone_dwell", dwell=2.0)
    assert router.route(short, now_s=10.0).reason == "short_dwell"
    assert router.route(enough, now_s=11.0).accepted


def test_generic_zone_enter_requires_dwell() -> None:
    router = AudioEventRouter(FakeSpeaker(), quiet_policy())
    item = event("water-enter", AudioEventKind.WATER, source_type="zone_enter")
    assert router.route(item, now_s=10.0).reason == "needs_dwell"


def test_stale_and_future_events_are_rejected() -> None:
    speaker = FakeSpeaker()
    router = AudioEventRouter(speaker, quiet_policy())
    old = event("old", AudioEventKind.DOOR_ENTER, timestamp=1_700_000_000.0)
    future = event("future", AudioEventKind.DOOR_ENTER, timestamp=1_700_000_200.0)
    assert router.route(old, now_s=10.0, wall_time_s=1_700_000_100.0).reason == "stale"
    assert router.route(future, now_s=11.0, wall_time_s=1_700_000_100.0).reason == "future"
    assert not speaker.calls


def test_identity_can_be_enriched_from_global_id() -> None:
    speaker = FakeSpeaker()
    router = AudioEventRouter(speaker, quiet_policy())
    item = event("enrich", AudioEventKind.DOOR_ENTER, person_id="", name="", gid=9)
    result = router.route(
        item,
        now_s=10.0,
        resolve_identity=lambda gid: ("NV09", "Lan") if gid == 9 else None,
    )
    assert result.accepted
    assert "Lan" in speaker.calls[-1][0]


def test_simultaneous_arrivals_become_one_group_line() -> None:
    speaker = FakeSpeaker()
    router = AudioEventRouter(speaker, quiet_policy())
    results = router.route_many([
        event("a", AudioEventKind.DOOR_ENTER, person_id="NV01", name="An", gid=1),
        event("b", AudioEventKind.DOOR_ENTER, person_id="NV02", name="Bình", gid=2),
    ], now_s=10.0)
    assert len(speaker.calls) == 1
    assert "mọi người" in speaker.calls[0][0].lower()
    assert any(result.accepted for result in results)


def test_critical_prewarm_excludes_slow_reminder_zones() -> None:
    texts = iter_prewarm_texts([("NV01", "Ngọc")], critical_only=True)
    assert any("Ngọc" in text for text in texts)
    assert not any("uống nước" in text for text in texts)
    assert not any("riêng tư" in text for text in texts)


def test_voice_turn_has_priority_over_zone_speech() -> None:
    speaker = FakeSpeaker()
    router = AudioEventRouter(
        speaker,
        quiet_policy(),
        turn_guard=lambda: True,
    )

    result = router.route(event("near-user", AudioEventKind.BE_XINH_NEAR), now_s=10.0)

    assert result.reason == "voice_turn"
    assert speaker.calls == []
