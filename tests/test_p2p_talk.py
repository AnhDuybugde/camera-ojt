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
    out = ImouP2PTalkOutput(
        ImouP2PCredentials(serial="S", password="P"), channel=2, volume=0.5
    )
    out(audio)
    assert seen["path"] == audio
    assert seen["channel"] == 2
    assert seen["volume"] == 0.5


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
