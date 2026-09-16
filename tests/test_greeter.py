"""VoiceGreeter: moi nguoi 1 loi chao trong cooldown (stub TTS/play)."""
from pathlib import Path
from typing import ClassVar

from camera_tracking.voice.greeter import VoiceGreeter
from camera_tracking.voice.zerotts_tts import add_display_name_aliases


class StubGreeter(VoiceGreeter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.played: list[str] = []
        self.synthesized: list[str] = []

    def _synthesize(self, text: str):
        self.synthesized.append(text)
        path = Path(self.cache_dir) / "stub.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-mp3")
        return path

    @staticmethod
    def _play(path) -> None:
        StubGreeter._played_paths.append(str(path))

    _played_paths: ClassVar[list[str]] = []


def _greeter(tmp_path) -> StubGreeter:
    StubGreeter._played_paths = []
    return StubGreeter(cache_dir=str(tmp_path), cooldown_s=60.0)


def test_known_greeted_once_per_cooldown(tmp_path) -> None:
    greeter = _greeter(tmp_path)
    assert greeter.wave_greet(day="2026-09-14", person_id="1",
                              display_name="Le Ho Anh Duy",
                              now_s=0.0) is True
    assert greeter.wave_greet(day="2026-09-14", person_id="1",
                              display_name="Le Ho Anh Duy",
                              now_s=10.0) is False
    assert greeter.wave_greet(day="2026-09-14", person_id="1",
                              display_name="Le Ho Anh Duy",
                              now_s=61.0) is True
    # Chi 2 loi chao duoc xep hang (lan 2 trong cooldown bi bo).
    assert greeter._queue.qsize() == 2


def test_unknown_gets_default_phrase(tmp_path) -> None:
    greeter = _greeter(tmp_path)
    assert greeter.wave_greet(day="2026-09-14", person_id=None,
                              display_name=None, now_s=0.0,
                              global_id=10) is True
    assert greeter.wave_greet(day="2026-09-14", person_id=None,
                              display_name=None, now_s=1.0,
                              global_id=11) is True
    assert greeter.wave_greet(day="2026-09-14", person_id=None,
                              display_name=None, now_s=1.0,
                              global_id=10) is False
    assert greeter._queue.qsize() == 2


def test_worker_plays_synthesized(tmp_path) -> None:
    greeter = _greeter(tmp_path)
    greeter.start()
    greeter.wave_greet(day="2026-09-14", person_id="2",
                       display_name="Le Van Dai", now_s=0.0)
    greeter.wave_greet(day="2026-09-14", person_id=None,
                       display_name=None, now_s=0.0)
    greeter._queue.join()
    assert greeter.synthesized == ["Xin chào Le Van Dai",
                                   "Xin chào quý khách"]
    assert len(StubGreeter._played_paths) == 2


def test_pregen_full_names_are_aliased_to_gallery_short_names(tmp_path) -> None:
    anh_duy = tmp_path / "LeHoAnhDuy.wav"
    van_dai = tmp_path / "LeVanDai.wav"
    aliases = add_display_name_aliases(
        {
            "Xin chào Lê Hồ Anh Duy": anh_duy,
            "Xin chào Lê Văn Đại": van_dai,
        },
        ["Anh Duy", "Van Dai"],
    )
    assert aliases["Xin chào Anh Duy"] == anh_duy
    assert aliases["Xin chào Van Dai"] == van_dai


def test_worker_calls_on_spoken_after_output(tmp_path) -> None:
    spoken: list[str] = []
    greeter = StubGreeter(
        cache_dir=str(tmp_path),
        cooldown_s=0,
        on_spoken=spoken.append,
    )
    greeter.start()
    assert greeter.wave_greet(
        day="2026-09-16", person_id="1", display_name="Anh Duy", now_s=0
    )
    greeter._queue.join()
    assert spoken == ["Xin chào Anh Duy"]
