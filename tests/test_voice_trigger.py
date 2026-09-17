"""Voice-trigger OR wave: cụm gọi lập trình sẵn (hello imou...)."""
from camera_tracking.voice.voice_trigger import (
    VoiceTrigger,
    is_voice_trigger,
    load_wake_phrases,
    normalize_trigger_text,
)


def test_normalize_strips_vietnamese_diacritics() -> None:
    assert normalize_trigger_text("Xin chào Quý Khách!") == "xin chao quy khach!"
    assert normalize_trigger_text("  HELLO...  ") == "hello..."


def test_keyword_matches_wake_phrases() -> None:
    assert is_voice_trigger("hello imou")
    assert is_voice_trigger("Xin chào Imou!")
    assert is_voice_trigger("xin chao imou")
    assert is_voice_trigger("a hello imou")
    # "hello" trần không còn là cụm gọi (chống TV/phòng ồn nhầm).
    assert not is_voice_trigger("hello")
    assert not is_voice_trigger("xin chào")
    # "camera" không còn là tiếng đệm.
    assert not is_voice_trigger("hello imou camera")
    # Tối đa 1 tiếng đệm.
    assert not is_voice_trigger("dạ xin chào imou ạ")
    assert not is_voice_trigger("tạp âm không tồn tại")
    assert not is_voice_trigger("")


def test_keyword_rejects_long_or_embedded_transcript() -> None:
    assert not is_voice_trigger("Tôi đang nói chuyện về hello imou trong cuộc họp")
    assert not is_voice_trigger("shelloworld")
    assert not is_voice_trigger("xin chào imou rồi chúng ta bắt đầu cuộc họp hôm nay")


def test_trigger_recent_window_and_consume() -> None:
    trigger = VoiceTrigger(window_s=5.0, inhibit_s=4.0)
    assert trigger.note_heard("Hello imou!", now=100.0) is True
    ok, text = trigger.recent(now=102.0)
    assert ok and text == "Hello imou!"
    # Hết window -> không greet nữa.
    ok, _ = trigger.recent(now=106.0)
    assert not ok
    # 1 cụm gọi chỉ greet 1 lần.
    trigger.note_heard("xin chào imou", now=200.0)
    trigger.consume(now=200.0)
    ok, _ = trigger.recent(now=201.0)
    assert not ok


def test_trigger_inhibit_blocks_self_echo() -> None:
    trigger = VoiceTrigger(window_s=5.0, inhibit_s=4.0)
    trigger.set_inhibit(now=50.0)
    # Loa TTS vừa phát -> mic nghe lại chính mình thì bỏ.
    assert trigger.note_heard("hello imou", now=51.0) is False
    ok, _ = trigger.recent(now=51.5)
    assert not ok
    # Hết inhibit -> nghe lại bình thường.
    assert trigger.note_heard("hello imou", now=55.0) is True


def test_trigger_uses_custom_fillers_and_max() -> None:
    trigger = VoiceTrigger(
        trigger_words=("hello imou",), fillers={"camera"}, max_fillers=2)
    assert trigger.note_heard("hello imou camera camera", now=1.0) is True
    assert trigger.note_heard("hello imou a", now=2.0) is False


def test_load_wake_phrases_from_folder(tmp_path) -> None:
    folder = tmp_path / "voice_triggers"
    folder.mkdir()
    (folder / "a.yaml").write_text(
        "phrases:\n  - ok imou\nfillers:\n  - oi\nmax_fillers: 1\n",
        encoding="utf-8",
    )
    (folder / "b.yml").write_text(
        "phrases:\n  - ok imou\n  - chào imou\n", encoding="utf-8")
    phrases, fillers, max_fill = load_wake_phrases(folder, extra_words=["hey imou"])
    assert phrases == ["ok imou", "chào imou", "hey imou"]
    assert "oi" in fillers and "a" in fillers
    assert max_fill == 1
    assert is_voice_trigger("ok imou", phrases, fillers, max_fill)


def test_load_wake_phrases_missing_dir_falls_back(tmp_path) -> None:
    phrases, _fillers, _max = load_wake_phrases(tmp_path / "khong-co")
    assert len(phrases) > 0
    assert is_voice_trigger("hello imou", phrases)
