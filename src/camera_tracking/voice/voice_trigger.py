"""Keyword gate cho voice-greet: chi chao khi nghe "hello" / "xin chao".

OR voi palm: giơ đủ bàn tay 5 ngón HOẶC nói hello/xin chào đều greet
(unknown + employee như nhau, còn lại không tự ý greet).

Chuẩn hoá không dấu để STT faster-whisper (vi) hay Vosk đều khớp:
"Xin chào" / "xin chao" / "HELLO!" đều trigger.
"""
from __future__ import annotations

import threading
import time
import unicodedata
import re


DEFAULT_TRIGGER_WORDS: tuple[str, ...] = ("hello", "xin chao")


def normalize_trigger_text(text: str) -> str:
    """Lowercase, bỏ dấu tiếng Việt, gom whitespace (để so khớp keyword)."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    # đ không tách được bằng NFD nên thay tay.
    stripped = stripped.replace("đ", "d")
    return " ".join(stripped.split())


def is_voice_trigger(text: str, trigger_words: tuple[str, ...] | list[str] = DEFAULT_TRIGGER_WORDS) -> bool:
    """Khớp câu chào ngắn, tránh substring/hallucination trong câu dài."""
    norm = normalize_trigger_text(text)
    if not norm:
        return False
    tokens = re.findall(r"[a-z0-9]+", norm)
    if not tokens or len(tokens) > 6:
        return False
    fillers = {"a", "da", "oi", "camera", "cam", "hey", "hi"}
    for word in trigger_words:
        word_tokens = re.findall(r"[a-z0-9]+", normalize_trigger_text(word))
        if not word_tokens:
            continue
        width = len(word_tokens)
        for start in range(len(tokens) - width + 1):
            if tokens[start:start + width] != word_tokens:
                continue
            remaining = tokens[:start] + tokens[start + width:]
            if all(token in fillers for token in remaining):
                return True
    return False


class VoiceTrigger:
    """Ghi nhớ lần nghe "hello/xin chào" gần nhất (thread-safe).

    Mic là toàn cục (không gắn được vào từng GID) nên main loop dùng
    ``recent()`` để greet người gần nhất, rồi ``consume()`` để 1 câu
    hello chỉ greet 1 lần. ``set_inhibit()`` chặn mic tự nghe loa TTS.
    """

    def __init__(
        self,
        trigger_words: tuple[str, ...] | list[str] = DEFAULT_TRIGGER_WORDS,
        window_s: float = 5.0,
        inhibit_s: float = 4.0,
    ) -> None:
        self.trigger_words = tuple(trigger_words) if trigger_words else DEFAULT_TRIGGER_WORDS
        self.window_s = max(0.0, float(window_s))
        self.inhibit_s = max(0.0, float(inhibit_s))
        self._lock = threading.Lock()
        self._last_heard: float = float("-inf")
        self._last_text: str | None = None
        self._inhibit_until: float = float("-inf")

    def note_heard(self, text: str, now: float | None = None) -> bool:
        """Ghi nhận 1 đoạn STT; True khi khớp keyword (đã qua inhibit)."""
        if not is_voice_trigger(text, self.trigger_words):
            return False
        at = time.monotonic() if now is None else now
        with self._lock:
            if at < self._inhibit_until:
                return False
            self._last_heard = at
            self._last_text = text.strip()
            return True

    def recent(self, now: float | None = None) -> tuple[bool, str | None]:
        """(True, text) khi có hello trong window và không bị inhibit."""
        at = time.monotonic() if now is None else now
        with self._lock:
            if at < self._inhibit_until:
                return False, None
            if at - self._last_heard <= self.window_s:
                return True, self._last_text
            return False, None

    def consume(self, now: float | None = None) -> None:
        """Đánh dấu đã dùng hello này (1 câu hello chỉ greet 1 lần)."""
        at = time.monotonic() if now is None else now
        with self._lock:
            self._last_heard = float("-inf")
            self._last_text = None
        self.set_inhibit(at)

    def set_inhibit(self, now: float | None = None) -> None:
        """Chặn trigger trong inhibit_s (tránh mic nghe chính loa TTS)."""
        at = time.monotonic() if now is None else now
        with self._lock:
            self._inhibit_until = at + self.inhibit_s

    def inhibited(self, now: float | None = None) -> bool:
        at = time.monotonic() if now is None else now
        with self._lock:
            return at < self._inhibit_until


__all__ = ["DEFAULT_TRIGGER_WORDS", "VoiceTrigger", "is_voice_trigger", "normalize_trigger_text"]
