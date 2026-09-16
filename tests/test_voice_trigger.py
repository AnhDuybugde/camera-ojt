"""Voice-trigger OR palm: hello/xin chào (không dấu vẫn khớp)."""
from camera_tracking.voice.voice_trigger import (
    VoiceTrigger,
    is_voice_trigger,
    normalize_trigger_text,
)


def test_normalize_strips_vietnamese_diacritics() -> None:
    assert normalize_trigger_text("Xin chào Quý Khách!") == "xin chao quy khach!"
    assert normalize_trigger_text("  HELLO...  ") == "hello..."


def test_keyword_matches_hello_and_xin_chao() -> None:
    assert is_voice_trigger("Hello camera!")
    assert is_voice_trigger("Xin chào camera!")
    assert is_voice_trigger("xin chao")
    assert is_voice_trigger("Dạ xin chào ạ")
    assert not is_voice_trigger("tạp âm không tồn tại")
    assert not is_voice_trigger("")


def test_keyword_rejects_long_or_embedded_transcript() -> None:
    assert not is_voice_trigger("Tôi đang nói chuyện về hello world trong cuộc họp")
    assert not is_voice_trigger("shelloworld")
    assert not is_voice_trigger("xin chào rồi chúng ta bắt đầu cuộc họp hôm nay")


def test_trigger_recent_window_and_consume() -> None:
    trigger = VoiceTrigger(window_s=5.0, inhibit_s=4.0)
    assert trigger.note_heard("Hello!", now=100.0) is True
    ok, text = trigger.recent(now=102.0)
    assert ok and text == "Hello!"
    # Hết window -> không greet nữa.
    ok, _ = trigger.recent(now=106.0)
    assert not ok
    # 1 câu hello chỉ greet 1 lần.
    trigger.note_heard("xin chào", now=200.0)
    trigger.consume(now=200.0)
    ok, _ = trigger.recent(now=201.0)
    assert not ok


def test_trigger_inhibit_blocks_self_echo() -> None:
    trigger = VoiceTrigger(window_s=5.0, inhibit_s=4.0)
    trigger.set_inhibit(now=50.0)
    # Loa TTS vừa phát -> mic nghe lại chính mình thì bỏ.
    assert trigger.note_heard("hello", now=51.0) is False
    ok, _ = trigger.recent(now=51.5)
    assert not ok
    # Hết inhibit -> nghe lại bình thường.
    assert trigger.note_heard("hello", now=55.0) is True
