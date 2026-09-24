from __future__ import annotations

import sys
import wave
from io import BytesIO
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from halinh_assistant import (  # noqa: E402
    active_voice_volume,
    gemini_model_candidates,
    is_gemini_capacity_error,
    pcm_to_wav_bytes,
)


def test_gemini_fallback_chain_is_ordered_and_unique() -> None:
    assert gemini_model_candidates(
        "gemini-3.5-flash", "gemini-3.5-flash-lite"
    ) == ["gemini-3.5-flash", "gemini-3.5-flash-lite"]
    assert gemini_model_candidates(
        "gemini-3.5-flash", "gemini-3.5-flash"
    ) == ["gemini-3.5-flash"]


def test_gemini_capacity_errors_use_fallback() -> None:
    assert is_gemini_capacity_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert is_gemini_capacity_error(RuntimeError("503 UNAVAILABLE"))
    assert not is_gemini_capacity_error(RuntimeError("400 INVALID_ARGUMENT"))


def test_pcm_to_wav_bytes_preserves_voice_format() -> None:
    pcm = b"\x01\x00\x02\x00" * 80
    payload = pcm_to_wav_bytes(pcm, sample_rate=16_000)

    with wave.open(BytesIO(payload), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.readframes(wav.getnframes()) == pcm


def test_voice_volume_uses_single_level_and_migrates_old_key(tmp_path) -> None:
    path = tmp_path / "volume.json"
    path.write_text(
        '{"percent":80}',
        encoding="utf-8",
    )

    assert active_voice_volume(path) == 0.8

    path.write_text('{"normal_percent":65}', encoding="utf-8")
    assert active_voice_volume(path) == 0.65
