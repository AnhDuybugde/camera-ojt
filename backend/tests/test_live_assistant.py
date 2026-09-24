from __future__ import annotations

import asyncio
from types import SimpleNamespace

from scripts.be_xinh_live_assistant import _send_audio_turn


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
