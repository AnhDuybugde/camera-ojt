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
    greeter.start()
    assert greeter.wave_greet(day="2026-09-14", person_id="1",
                              display_name="Le Ho Anh Duy",
                              now_s=0.0) is True
    greeter._queue.join()
    assert greeter.wave_greet(day="2026-09-14", person_id="1",
                              display_name="Le Ho Anh Duy",
                              now_s=10.0) is False
    assert greeter.wave_greet(day="2026-09-14", person_id="1",
                              display_name="Le Ho Anh Duy",
                              now_s=61.0) is True
    greeter._queue.join()
    # Chi 2 loi chao duoc phat (lan 2 trong cooldown bi bo).
    assert len(StubGreeter._played_paths) == 2


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
    StubGreeter._played_paths = []
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


def test_channel_gives_independent_cooldown_per_speaker(tmp_path) -> None:
    greeter = _greeter(tmp_path)
    greeter.start()
    assert greeter.wave_greet(day="2026-09-16", person_id="1",
                              display_name="Anh Duy",
                              now_s=0.0, channel=1) is True
    greeter._queue.join()
    # Cung nguoi nhung loa khac (cua B) van duoc chao.
    assert greeter.wave_greet(day="2026-09-16", person_id="1",
                              display_name="Anh Duy",
                              now_s=1.0, channel=2) is True
    greeter._queue.join()
    # Cung loa thi van cooldown.
    assert greeter.wave_greet(day="2026-09-16", person_id="1",
                              display_name="Anh Duy",
                              now_s=2.0, channel=1) is False
    assert greeter.cooldown_remaining(day="2026-09-16", person_id="1",
                                      now_s=2.0, channel=2) > 0


def test_same_person_collapses_to_single_queued_turn(tmp_path) -> None:
    # Vay lien tuc khi loa ban: chi giu 1 luot cho, khong doi 5 cuc.
    greeter = _greeter(tmp_path)
    assert greeter.wave_greet(day="2026-09-16", person_id="1",
                              display_name="Anh Duy",
                              now_s=0.0, channel=1) is True
    assert greeter.wave_greet(day="2026-09-16", person_id="1",
                              display_name="Anh Duy",
                              now_s=0.5, channel=1) is False
    assert greeter._queue.qsize() == 1


def test_proactive_skips_cooldown_but_respects_quiet_window(tmp_path) -> None:
    greeter = StubGreeter(cache_dir=str(tmp_path), cooldown_s=60.0,
                          proactive_quiet_s=5.0)
    StubGreeter._played_paths = []
    greeter.start()
    # Tu dong truoc: nap cooldown 60s.
    assert greeter.face_greet(day="2026-09-16", person_id="1",
                              display_name="Anh Duy", now_s=0.0) is True
    greeter._queue.join()
    # Chu dong sau khoang lang: khong bi cooldown 60s chan.
    assert greeter.face_greet(day="2026-09-16", person_id="1",
                              display_name="Anh Duy", now_s=6.0,
                              proactive=True) is True
    greeter._queue.join()
    # Vay tiep trong khoang lang 5s ke tu lan phat: bi bo qua (chong spam).
    assert greeter.face_greet(day="2026-09-16", person_id="1",
                              display_name="Anh Duy", now_s=7.0,
                              proactive=True) is False
    # Qua khoang lang: nhan lai.
    assert greeter.face_greet(day="2026-09-16", person_id="1",
                              display_name="Anh Duy", now_s=12.0,
                              proactive=True) is True
    greeter._queue.join()


def test_proactive_upgrades_pending_auto_request(tmp_path) -> None:
    greeter = StubGreeter(cache_dir=str(tmp_path), cooldown_s=60.0)
    # Tu dong xep cho truoc (loa dang ban, chua phat).
    assert greeter.face_greet(day="2026-09-16", person_id="2",
                              display_name="Van Dai", now_s=0.0) is True
    # Nguoi do vẫy tay: nang len chu dong, van chi 1 luot.
    assert greeter.face_greet(day="2026-09-16", person_id="2",
                              display_name="Van Dai", now_s=0.5,
                              proactive=True) is True
    assert greeter._queue.qsize() == 2  # cu + moi (cu se bi huy khi lay ra)
    StubGreeter._played_paths = []
    greeter.start()
    greeter._queue.join()
    # Chi phat 1 lan (luot tu dong cu bi huy).
    assert len(StubGreeter._played_paths) == 1


def test_proactive_plays_before_earlier_auto(tmp_path) -> None:
    outputs: list[str] = []
    greeter = StubGreeter(cache_dir=str(tmp_path), cooldown_s=0)
    greeter.output = lambda path, channel=None: outputs.append(Path(path).name)
    # Tu dong den truoc, chu dong den sau.
    assert greeter.face_greet(day="2026-09-16", person_id="8",
                              display_name="Nhi", now_s=0.0) is True
    assert greeter.face_greet(day="2026-09-16", person_id="9",
                              display_name="Phuc", now_s=0.1,
                              proactive=True) is True
    StubGreeter._played_paths = []
    greeter.start()
    greeter._queue.join()
    assert outputs == ["stub.mp3", "stub.mp3"]
    assert greeter.synthesized[0] == "Xin chào Phuc"


def test_queued_greeting_carries_speaker_channel(tmp_path) -> None:
    outputs: list[tuple] = []
    greeter = StubGreeter(cache_dir=str(tmp_path), cooldown_s=0)
    greeter.output = lambda path, channel=None: outputs.append((str(path), channel))
    StubGreeter._played_paths = []
    greeter.start()
    assert greeter.face_greet(day="2026-09-16", person_id=None,
                              display_name=None, now_s=0.0,
                              global_id=7, channel=2)
    greeter._queue.join()
    assert outputs and outputs[0][1] == 2


def test_legacy_single_arg_output_still_works(tmp_path) -> None:
    played: list[str] = []
    greeter = StubGreeter(cache_dir=str(tmp_path), cooldown_s=0)
    greeter.output = lambda path: played.append(str(path))
    StubGreeter._played_paths = []
    greeter.start()
    assert greeter.face_greet(day="2026-09-16", person_id="9",
                              display_name="Nhi", now_s=0.0,
                              global_id=3, channel=2)
    greeter._queue.join()
    assert len(played) == 1


def test_worker_calls_on_spoken_after_output(tmp_path) -> None:
    spoken: list[str] = []
    greeter = StubGreeter(
        cache_dir=str(tmp_path),
        cooldown_s=0,
        on_spoken=spoken.append,
    )
    StubGreeter._played_paths = []
    greeter.start()
    assert greeter.wave_greet(
        day="2026-09-16", person_id="1", display_name="Anh Duy", now_s=0
    )
    greeter._queue.join()
    assert spoken == ["Xin chào Anh Duy"]


def test_cancel_pending_unknown_before_name_greet(tmp_path) -> None:
    # "quy khach" dang cho -> mat xac nhan -> huy, chi chao ten.
    played: list[str] = []
    greeter = StubGreeter(
        cache_dir=str(tmp_path), cooldown_s=0, unknown_cooldown_s=60.0)
    greeter.output = lambda path, channel=None: played.append(str(path))
    greeter.start()
    assert greeter.face_greet(day="2026-09-17", person_id=None,
                              display_name=None, now_s=0.0,
                              global_id=7, channel=2) is True
    assert greeter.cancel_pending_unknown(global_id=7) is True
    assert greeter.cancel_pending_unknown(global_id=7) is False
    assert greeter.face_greet(day="2026-09-17", person_id="AnhDuy",
                              display_name="Anh Duy", now_s=0.5,
                              global_id=7, channel=2) is True
    greeter._queue.join()
    assert greeter.synthesized == ["Xin chào Anh Duy"]


def test_unknown_spoken_age_guards_double_greeting(tmp_path) -> None:
    greeter = StubGreeter(
        cache_dir=str(tmp_path), cooldown_s=0, unknown_cooldown_s=10.0)
    greeter.start()
    assert greeter.unknown_spoken_age(
        day="2026-09-17", global_id=7, channel=2, now_s=100.0) is None
    assert greeter.face_greet(day="2026-09-17", person_id=None,
                              display_name=None, now_s=100.0,
                              global_id=7, channel=2) is True
    greeter._queue.join()
    age = greeter.unknown_spoken_age(
        day="2026-09-17", global_id=7, channel=2, now_s=105.0)
    assert age is not None and 0 <= age < 10.0
    # GID khac khong bi anh huong.
    assert greeter.unknown_spoken_age(
        day="2026-09-17", global_id=8, channel=2, now_s=105.0) is None
