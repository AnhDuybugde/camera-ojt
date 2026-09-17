"""Keyword gate cho voice-greet: chi chao khi nghe cum goi lap trinh san.

OR voi wave: vẫy tay HOẶC nói cụm gọi (mặc định "hello imou" /
"xin chào imou" trong config/voice_triggers/) đều greet.

Cụm gọi cụ thể ("... imou") thay cho "hello" trần để TV/phòng ồn không
gây trigger nhầm. Chuẩn hoá không dấu để STT faster-whisper (vi) khớp:
"Xin chào Imou" / "xin chao imou" / "HELLO IMOU!" đều trigger.
"""
from __future__ import annotations

import threading
import time
import unicodedata
import re
from pathlib import Path


DEFAULT_TRIGGER_WORDS: tuple[str, ...] = ("hello imou", "xin chao imou")

# Tiếng đệm cho phép đứng kèm cụm gọi (đã bỏ "camera"/"cam" chống nhầm).
DEFAULT_FILLERS: frozenset[str] = frozenset({"a", "da", "oi", "hey", "hi"})
# Cụm gọi đứng gần như một mình: tối đa 1 tiếng đệm.
DEFAULT_MAX_FILLERS: int = 1


def normalize_trigger_text(text: str) -> str:
    """Lowercase, bỏ dấu tiếng Việt, gom whitespace (để so khớp keyword)."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    # đ không tách được bằng NFD nên thay tay.
    stripped = stripped.replace("đ", "d")
    return " ".join(stripped.split())


def is_voice_trigger(
    text: str,
    trigger_words: tuple[str, ...] | list[str] = DEFAULT_TRIGGER_WORDS,
    fillers: frozenset[str] | set[str] | tuple[str, ...] | list[str] = DEFAULT_FILLERS,
    max_fillers: int = DEFAULT_MAX_FILLERS,
) -> bool:
    """Khớp cụm gọi đứng gần như một mình, tránh hallucination câu dài."""
    norm = normalize_trigger_text(text)
    if not norm:
        return False
    tokens = re.findall(r"[a-z0-9]+", norm)
    if not tokens or len(tokens) > 6:
        return False
    filler_set = set(fillers) if fillers else set()
    for word in trigger_words:
        word_tokens = re.findall(r"[a-z0-9]+", normalize_trigger_text(word))
        if not word_tokens:
            continue
        width = len(word_tokens)
        for start in range(len(tokens) - width + 1):
            if tokens[start:start + width] != word_tokens:
                continue
            remaining = tokens[:start] + tokens[start + width:]
            if (len(remaining) <= max(0, max_fillers)
                    and all(token in filler_set for token in remaining)):
                return True
    return False


def load_wake_phrases(
    trigger_dir: str | Path,
    extra_words: tuple[str, ...] | list[str] = (),
) -> tuple[list[str], set[str], int]:
    """Đọc mọi *.yaml trong folder cụm gọi -> (phrases, fillers, max_fillers).

    Muốn thêm cụm gọi mới ("ok imou", ...) chỉ cần thêm file yaml vào
    folder rồi restart pipeline. `extra_words` (từ voice_trigger_words
    trong config) được nối thêm. Không bao giờ raise: folder thiếu/trống
    thì trả default.
    """
    phrases: list[str] = []
    fillers: set[str] = set(DEFAULT_FILLERS)
    max_fillers = DEFAULT_MAX_FILLERS
    try:
        files = sorted(Path(trigger_dir).glob("*.yaml"))
        files += sorted(Path(trigger_dir).glob("*.yml"))
    except OSError:
        files = []
    for path in files:
        try:
            import yaml
        except ImportError:
            break
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for phrase in data.get("phrases") or []:
            cleaned = str(phrase).strip()
            if cleaned and cleaned not in phrases:
                phrases.append(cleaned)
        for filler in data.get("fillers") or []:
            cleaned = str(filler).strip().lower()
            if cleaned:
                fillers.add(cleaned)
        try:
            file_max = int(data.get("max_fillers", max_fillers))
        except (TypeError, ValueError):
            continue
        max_fillers = min(max_fillers, max(0, file_max))
    for word in extra_words or ():
        cleaned = str(word).strip()
        if cleaned and cleaned not in phrases:
            phrases.append(cleaned)
    if not phrases:
        phrases = list(DEFAULT_TRIGGER_WORDS)
    return phrases, fillers, max_fillers


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
        fillers: frozenset[str] | set[str] | tuple[str, ...] | list[str] = DEFAULT_FILLERS,
        max_fillers: int = DEFAULT_MAX_FILLERS,
    ) -> None:
        self.trigger_words = tuple(trigger_words) if trigger_words else DEFAULT_TRIGGER_WORDS
        self.fillers = set(fillers) if fillers else set(DEFAULT_FILLERS)
        self.max_fillers = max(0, int(max_fillers))
        self.window_s = max(0.0, float(window_s))
        self.inhibit_s = max(0.0, float(inhibit_s))
        self._lock = threading.Lock()
        self._last_heard: float = float("-inf")
        self._last_text: str | None = None
        self._inhibit_until: float = float("-inf")

    def note_heard(self, text: str, now: float | None = None) -> bool:
        """Ghi nhận 1 đoạn STT; True khi khớp cụm gọi (đã qua inhibit)."""
        if not is_voice_trigger(text, self.trigger_words, self.fillers, self.max_fillers):
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


__all__ = [
    "DEFAULT_FILLERS",
    "DEFAULT_MAX_FILLERS",
    "DEFAULT_TRIGGER_WORDS",
    "VoiceTrigger",
    "is_voice_trigger",
    "load_wake_phrases",
    "normalize_trigger_text",
]
