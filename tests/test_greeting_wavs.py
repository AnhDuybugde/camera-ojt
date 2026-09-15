"""WAV chao tao san (ZeroTTS): manifest, noi gapless, greeter uu tien."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from camera_tracking.voice.greeter import VoiceGreeter
from camera_tracking.voice.zerotts_tts import (
    concat_wavs_gapless,
    load_phrase_files,
)

soundfile = pytest.importorskip("soundfile")

SR = 48000


def _sine(path: Path, freq: float, seconds: float, sample_rate: int = SR) -> Path:
    samples = np.sin(
        2 * np.pi * freq * np.arange(int(seconds * sample_rate)) / sample_rate
    ).astype(np.float32)
    soundfile.write(str(path), samples, sample_rate)
    return path


def test_load_phrase_files_skips_missing_and_empty(tmp_path) -> None:
    ok = _sine(tmp_path / "unknown.wav", 440.0, 0.2)
    (tmp_path / "empty.wav").write_bytes(b"")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "voice": "maichi",
                "phrases": {
                    "Xin chào quý khách": "unknown.wav",
                    "Xin chào Ghost": "ghost.wav",
                    "Empty": "empty.wav",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    files = load_phrase_files(tmp_path)
    assert files == {"Xin chào quý khách": ok}


def test_load_phrase_files_no_manifest(tmp_path) -> None:
    assert load_phrase_files(tmp_path) == {}


def test_concat_gapless_is_sample_exact(tmp_path) -> None:
    first = _sine(tmp_path / "xin-chao.wav", 440.0, 0.3)
    second = _sine(tmp_path / "ten.wav", 660.0, 0.4)
    out = concat_wavs_gapless([first, second], tmp_path / "full.wav")
    merged, rate = soundfile.read(str(out), dtype="float32", always_2d=False)
    left, _ = soundfile.read(str(first), dtype="float32", always_2d=False)
    right, _ = soundfile.read(str(second), dtype="float32", always_2d=False)
    assert rate == SR
    # Khong chen silence/mau thua: do dai = tong, noi dung ghep khop tuyet doi.
    assert len(merged) == len(left) + len(right)
    np.testing.assert_allclose(merged[: len(left)], left, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(merged[len(left):], right, rtol=1e-5, atol=1e-6)


def test_concat_rejects_sample_rate_mismatch(tmp_path) -> None:
    first = _sine(tmp_path / "a.wav", 440.0, 0.2, sample_rate=48000)
    second = _sine(tmp_path / "b.wav", 440.0, 0.2, sample_rate=16000)
    with pytest.raises(ValueError, match="Sample rate mismatch"):
        concat_wavs_gapless([first, second], tmp_path / "out.wav")


def test_greeter_prefers_pregen_over_tts(tmp_path) -> None:
    wav = _sine(tmp_path / "LeHoAnhDuy.wav", 440.0, 0.2)
    greeter = VoiceGreeter(
        cache_dir=str(tmp_path / "cache"),
        phrase_files={"Xin chào Lê Hồ Anh Duy": wav},
    )
    greeter._tts_failed = True  # chung minh khong can TTS runtime
    assert greeter._synthesize("Xin chào Lê Hồ Anh Duy") == wav


def test_greeter_falls_back_when_pregen_missing(tmp_path) -> None:
    greeter = VoiceGreeter(
        cache_dir=str(tmp_path / "cache"),
        phrase_files={"Xin chào Ghost": tmp_path / "ghost.wav"},
    )
    greeter._tts_failed = True
    assert greeter._synthesize("Xin chào Ghost") is None


def test_config_greeting_defaults() -> None:
    from camera_tracking.config import load_config

    config = load_config(Path("config/default.yaml"))
    assert config.voice.greeting_dir == Path("output/voice_greetings")
    assert config.voice.zerotts_voice == "maichi"
    assert config.voice.zerotts_model == "zeroweight-ai/ZeroTTS"
