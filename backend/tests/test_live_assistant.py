from __future__ import annotations

import asyncio
from types import SimpleNamespace

from scripts import be_xinh_live_assistant as live
from scripts.be_xinh_live_assistant import (
    _build_live_config,
    _receive_turn,
    _send_audio_turn,
)


def test_native_live_config_uses_selected_voice() -> None:
    config = _build_live_config("native", "Leda")
    selected = config["speech_config"]["voice_config"]["prebuilt_voice_config"]

    assert config["response_modalities"] == ["AUDIO"]
    assert selected["voice_name"] == "Leda"
    assert config["speech_config"]["language_code"] == "vi-VN"
    assert config["system_instruction"]


def test_zerotts_live_config_keeps_audio_for_output_transcript() -> None:
    config = _build_live_config("zerotts", "Leda")

    assert config["response_modalities"] == ["AUDIO"]
    assert config["output_audio_transcription"] == {}
    assert config["input_audio_transcription"]["language_codes"] == ["vi-VN"]
    assert "Bé Xinh" in config["input_audio_transcription"]["custom_vocabulary"]


def test_presegmented_audio_uses_explicit_activity_boundaries() -> None:
    calls = []

    class _Session:
        async def send_realtime_input(self, **kwargs):
            calls.append(kwargs)

    class _Types:
        class ActivityStart:
            pass

        class ActivityEnd:
            pass

        @staticmethod
        def Blob(**kwargs):
            return SimpleNamespace(**kwargs)

    asyncio.run(_send_audio_turn(_Session(), b"\x01\x00" * 3200, _Types))

    assert isinstance(calls[0]["activity_start"], _Types.ActivityStart)
    assert isinstance(calls[-1]["activity_end"], _Types.ActivityEnd)
    assert all("audio_stream_end" not in call for call in calls)
    assert sum("audio" in call for call in calls) >= 1


def test_tool_result_waits_for_final_spoken_answer(monkeypatch) -> None:
    monkeypatch.setitem(live.TOOL_FUNCS, "demo_tool", lambda: "kết quả thật")

    tool_response = SimpleNamespace(
        data=None,
        tool_call=SimpleNamespace(function_calls=[SimpleNamespace(
            id="call-1", name="demo_tool", args={},
        )]),
        server_content=SimpleNamespace(
            input_transcription=None,
            output_transcription=None,
            model_turn=None,
            interrupted=False,
            turn_complete=True,
        ),
    )
    answer_response = SimpleNamespace(
        data=None,
        tool_call=None,
        server_content=SimpleNamespace(
            input_transcription=None,
            output_transcription=SimpleNamespace(text="Đây là kết quả thật."),
            model_turn=None,
            interrupted=False,
            turn_complete=True,
        ),
    )

    class _Session:
        def __init__(self):
            self.responses = [[tool_response], [answer_response]]
            self.sent = []

        async def receive(self):
            for item in self.responses.pop(0):
                yield item

        async def send_tool_response(self, **kwargs):
            self.sent.append(kwargs)

    class _Types:
        @staticmethod
        def FunctionResponse(**kwargs):
            return SimpleNamespace(**kwargs)

    session = _Session()
    _, answer, _ = asyncio.run(_receive_turn(
        session, SimpleNamespace(), _Types, stream_native_audio=False,
    ))

    assert session.sent
    assert answer == "Đây là kết quả thật."
