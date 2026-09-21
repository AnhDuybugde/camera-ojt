"""Backend P2P VisualTalk (thay imou_web): credentials + convert + output."""
from __future__ import annotations

from pathlib import Path

import pytest

from camera_tracking.config import load_config


def _clear_env(monkeypatch):
    for name in (
        "IMOU_DEVICE_ID",
        "IMOU_CAMERA_PASSWORD",
        "IMOU_PASSWORD",
        "IMOU_DEVICE_CODE",
        "IMOU_DEVICE_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)


def test_p2p_credentials_prefers_camera_password(monkeypatch):
    from camera_tracking.voice.p2p_talk import ImouP2PCredentials

    _clear_env(monkeypatch)
    monkeypatch.setenv("IMOU_DEVICE_ID", "SERIAL123")
    monkeypatch.setenv("IMOU_PASSWORD", "rtsp-pwd")
    monkeypatch.setenv("IMOU_DEVICE_CODE", "safety")
    monkeypatch.setenv("IMOU_CAMERA_PASSWORD", "dedicated")
    creds = ImouP2PCredentials.from_env()
    assert creds.serial == "SERIAL123"
    assert creds.password == "dedicated"


def test_p2p_credentials_fallback_device_code(monkeypatch):
    from camera_tracking.voice.p2p_talk import ImouP2PCredentials

    _clear_env(monkeypatch)
    monkeypatch.setenv("IMOU_DEVICE_ID", "SERIAL123")
    monkeypatch.setenv("IMOU_DEVICE_CODE", "safety-code")
    creds = ImouP2PCredentials.from_env()
    assert creds.password == "safety-code"


def test_p2p_credentials_missing(monkeypatch):
    from camera_tracking.voice.p2p_talk import ImouP2PCredentials, P2PTalkError

    _clear_env(monkeypatch)
    with pytest.raises(P2PTalkError):
        ImouP2PCredentials.from_env()


def test_p2p_output_calls_send(monkeypatch, tmp_path):
    from camera_tracking.voice import p2p_talk
    from camera_tracking.voice.p2p_talk import ImouP2PCredentials, ImouP2PTalkOutput

    audio = tmp_path / "hello.mp3"
    audio.write_bytes(b"fake")
    seen = {}

    def _fake_send(path, creds, **kwargs):
        seen["path"] = Path(path)
        seen["channel"] = kwargs.get("channel")
        seen["volume"] = kwargs.get("volume")

    monkeypatch.setattr(p2p_talk, "send_audio_file", _fake_send)
    monkeypatch.setattr(
        ImouP2PTalkOutput, "_convert_cached", lambda self, path: b"aac")
    out = ImouP2PTalkOutput(
        ImouP2PCredentials(serial="S", password="P"), channel=2, volume=0.5,
        persistent=False,
    )
    out(audio)
    assert seen["path"] == audio
    assert seen["channel"] == 2
    assert seen["volume"] == 0.5


def test_p2p_output_channel_override_per_greeting(monkeypatch, tmp_path):
    from camera_tracking.voice import p2p_talk
    from camera_tracking.voice.p2p_talk import ImouP2PCredentials, ImouP2PTalkOutput

    audio = tmp_path / "hello.mp3"
    audio.write_bytes(b"fake")
    seen = {}

    def _fake_send(path, creds, **kwargs):
        seen["channel"] = kwargs.get("channel")

    monkeypatch.setattr(p2p_talk, "send_audio_file", _fake_send)
    monkeypatch.setattr(
        ImouP2PTalkOutput, "_convert_cached", lambda self, path: b"aac")
    out = ImouP2PTalkOutput(
        ImouP2PCredentials(serial="S", password="P"), channel=1, volume=0.5,
        persistent=False,
    )
    out(audio, 2)
    assert seen["channel"] == 2
    out(audio)
    assert seen["channel"] == 1


def test_voice_config_p2p_speaker_channels():
    config = load_config(Path("config/default.yaml"))
    assert config.voice.p2p_channel == 1
    assert config.voice.p2p_channel_b == 2


def test_output_caches_aac_conversion(monkeypatch, tmp_path):
    from camera_tracking.voice import p2p_talk
    from camera_tracking.voice.p2p_talk import ImouP2PCredentials, ImouP2PTalkOutput

    audio = tmp_path / "hello.mp3"
    audio.write_bytes(b"fake")
    calls = []

    def _fake_convert(path, *, sample_rate, volume):
        calls.append(str(path))
        return b"aac-bytes"

    monkeypatch.setattr(p2p_talk, "convert_file_to_aac_adts", _fake_convert)
    out = ImouP2PTalkOutput(
        ImouP2PCredentials(serial="S", password="P"), persistent=False)
    assert out._convert_cached(audio) == b"aac-bytes"
    assert out._convert_cached(audio) == b"aac-bytes"
    assert len(calls) == 1


def test_persistent_tunnel_reuses_and_recovers(monkeypatch):
    from camera_tracking.voice import p2p_talk
    from camera_tracking.voice.p2p_talk import (
        ImouP2PCredentials, ImouP2PTalkOutput, P2PTalkError)

    handshakes = []
    plays = []

    class _FakePtcp:
        def close(self):
            pass

    class _FakeTunnel:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self, host, port):
            import asyncio as _aio

            await _aio.Event().wait()

    async def _fake_handshake(*args, **kwargs):
        handshakes.append(1)
        return _FakePtcp()

    def _fake_talk(aac, creds, host, port, *, channel, timeout, sample_rate):
        plays.append(channel)

    monkeypatch.setattr(p2p_talk, "p2p_handshake", _fake_handshake)
    monkeypatch.setattr(p2p_talk, "DHP2PTunnel", _FakeTunnel)
    monkeypatch.setattr(p2p_talk, "_visualtalk_once", _fake_talk)
    out = ImouP2PTalkOutput(
        ImouP2PCredentials(serial="S", password="P"), persistent=True)
    out._convert_cached = lambda path: b"aac"
    out.warmup()
    out(__import__("pathlib").Path("a.mp3"), 1)
    out(__import__("pathlib").Path("b.mp3"), 2)
    # 1 lan bat tay cho ca warmup + 2 lan phat.
    assert len(handshakes) == 1
    assert plays == [1, 2]
    out.close()

    # Hong tunnel -> tu bat tay lai o lan sau.
    out2 = ImouP2PTalkOutput(
        ImouP2PCredentials(serial="S", password="P"), persistent=True)
    out2._convert_cached = lambda path: b"aac"
    out2._tunnel._up = True
    out2._tunnel._server_task = None
    calls = []

    def _flaky(aac, creds, host, port, *, channel, timeout, sample_rate):
        calls.append(channel)
        if len(calls) == 1:
            raise P2PTalkError("boom")

    monkeypatch.setattr(p2p_talk, "_visualtalk_once", _flaky)
    # ensure() thay tunnel sup (task None) -> bat tay moi truoc khi phat,
    # hong lan dau -> bat tay lai + thu lan 2.
    before = len(handshakes)
    out2(__import__("pathlib").Path("c.mp3"), 2)
    assert len(handshakes) == before + 2
    assert calls == [2, 2]
    out2.close()


def test_convert_missing_file(tmp_path):
    from camera_tracking.voice.p2p_talk import P2PTalkError, convert_file_to_aac_adts

    with pytest.raises(P2PTalkError):
        convert_file_to_aac_adts(tmp_path / "nope.mp3")


def test_voice_config_p2p_defaults():
    config = load_config(Path("config/default.yaml"))
    assert config.voice.backend == "imou_p2p"
    assert config.voice.p2p_channel == 1
    assert config.voice.p2p_sample_rate == 16000
    assert 0.0 <= config.voice.p2p_volume <= 1.0


def test_vendored_self_tests():
    from camera_tracking.voice.p2p.imou_dhav import self_test as dhav_self_test
    from camera_tracking.voice.p2p.imou_dhp2p import self_test as dhp2p_self_test
    from camera_tracking.voice.p2p.imou_wsse import self_test as wsse_self_test

    dhav_self_test()
    dhp2p_self_test()
    wsse_self_test()
