from camera_tracking.audio.announcer import CameraCheckInAnnouncer
from pathlib import Path
import pytest

from camera_tracking.audio.backchannel import (
    CameraAudioError,
    _backchannel_track,
    _control_uri,
    _interleaved_data_channel,
)


def test_backchannel_track_selects_sendonly_audio() -> None:
    sdp = """v=0
m=video 0 RTP/AVP 98
a=rtpmap:98 H265/90000
a=control:trackID=0
a=recvonly
m=audio 0 RTP/AVP 97
a=rtpmap:97 MPEG4-GENERIC/16000
a=control:trackID=1
a=recvonly
m=audio 0 RTP/AVP 104
a=rtpmap:104 MPEG4-GENERIC/16000
a=control:trackID=5
a=sendonly
"""

    assert _backchannel_track(sdp) == ("trackID=5", 104, 16_000)


def test_control_uri_uses_content_base() -> None:
    assert _control_uri(
        "rtsp://camera/live",
        "rtsp://camera/profile/",
        "trackID=5",
    ) == "rtsp://camera/profile/trackID=5"


def test_interleaved_data_channel_uses_setup_response() -> None:
    assert _interleaved_data_channel(
        "RTP/AVP/TCP;unicast;destination=192.0.2.1;interleaved=6-7"
    ) == 6


def test_interleaved_data_channel_requires_tcp_channel() -> None:
    with pytest.raises(CameraAudioError, match="interleaved"):
        _interleaved_data_channel("RTP/AVP;unicast;client_port=10000-10001")


def test_announcer_is_unavailable_without_camera_credentials(monkeypatch) -> None:
    for key in ("IMOU_IP", "IMOU_USER", "IMOU_PASSWORD"):
        monkeypatch.delenv(key, raising=False)

    assert CameraCheckInAnnouncer.from_env() is None


def test_announcer_ignores_legacy_enabled_flag(monkeypatch, tmp_path) -> None:
    helper = tmp_path / "talk_helper.py"
    helper.write_text("", encoding="utf-8")
    monkeypatch.setenv("CAMERA_TTS_ENABLED", "false")
    monkeypatch.setenv("IMOU_IP", "127.0.0.1")
    monkeypatch.setenv("IMOU_USER", "test")
    monkeypatch.setenv("IMOU_PASSWORD", "test")
    monkeypatch.setenv("IMOU_TALK_HELPER", str(helper))
    monkeypatch.setenv("HAMY_AUDIO_CACHE_DIR", str(tmp_path / "cache"))

    announcer = CameraCheckInAnnouncer.from_env()
    assert announcer is not None
    announcer.close()


def test_announcer_can_be_muted_without_stopping_worker() -> None:
    announcer = CameraCheckInAnnouncer(
        host="127.0.0.1",
        username="test",
        password="test",
        helper_path=Path(__file__),
        cache_dir=Path("output/hamy_audio_cache"),
    )
    try:
        muted = announcer.set_muted(True)
        assert muted["muted"] is True
        assert muted["enabled"] is False
        assert announcer.say("Không được phát") is False

        enabled = announcer.set_muted(False)
        assert enabled["muted"] is False
        assert enabled["enabled"] is True
    finally:
        announcer.close()
