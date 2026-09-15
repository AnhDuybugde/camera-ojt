"""Chao bang giong noi khi co nguoi vay tay (khong block inference).

- Hang doi + thread rieng phat am thanh; uu tien WAV tao san bang ZeroTTS
  (scripts/build_greeting_wavs.py, 1 cau = 1 file -> phat 1 session P2P
  khong ngat quang), fallback TTS edge-tts vi-VN cache mp3 theo cau.
- Phat bang ffplay (di kem ffmpeg, khong them dep) khi backend local.
- Moi nguoi cooldown rieng (mac dinh 60s) ke ca vay lien tuc.
"""
from __future__ import annotations

import asyncio
import hashlib
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


def _phrase_key(text: str, voice: str) -> str:
    digest = hashlib.sha256(f"{voice}|{text}".encode()).hexdigest()[:16]
    return f"{digest}.mp3"


@dataclass
class VoiceGreeter:
    """Phat 'Xin chao <ten>' (quen) / 'Quen khong ma chao' (la)."""

    cache_dir: str | Path = "output/voice_cache"
    voice: str = "vi-VN-HoaiMyNeural"
    cooldown_s: float = 60.0
    unknown_phrase: str = "Xin chào quý khách"
    max_queue_age_s: float = 5.0
    output: Callable[[Path], None] | None = None
    # WAV tao san (ZeroTTS): {cau chao dung nguyen van: duong dan file}.
    # Khop thi dung ngay, khong goi edge-tts (offline hoan toan).
    phrase_files: dict[str, str | Path] = field(default_factory=dict)
    _last_spoken: dict[tuple[str, str], float] = field(
        default_factory=dict, init=False)
    _queue: queue.Queue = field(default_factory=queue.Queue, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _tts_failed: bool = field(default=False, init=False)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._drain, name="voice-greeter", daemon=True)
        self._thread.start()

    def wave_greet(
        self,
        *,
        day: str,
        person_id: str | None,
        display_name: str | None,
        now_s: float,
        global_id: int | None = None,
    ) -> bool:
        """Xep 1 loi chao neu qua cooldown. Tra True khi da xep."""
        identity_key = person_id or f"unknown:{global_id if global_id is not None else 'any'}"
        key = (day, identity_key)
        if now_s - self._last_spoken.get(key, float("-inf")) < self.cooldown_s:
            return False
        self._last_spoken[key] = now_s
        if person_id and display_name:
            text = f"Xin chào {display_name}"
        else:
            text = self.unknown_phrase
        self._queue.put((text, time.monotonic()))
        return True

    def face_greet(
        self,
        *,
        day: str,
        person_id: str | None,
        display_name: str | None,
        now_s: float,
        global_id: int | None = None,
    ) -> bool:
        """Chao khi nhan dien mat (dung truoc camera), khong can vay tay.

        Dung chung cooldown/queue voi wave_greet de khong spam loa khi
        dung lau truoc camera. Tra True khi da xep hang phat.
        """
        return self.wave_greet(
            day=day,
            person_id=person_id,
            display_name=display_name,
            now_s=now_s,
            global_id=global_id,
        )

    def prewarm(self, phrases: list[str]) -> None:
        """Tao cache TTS nen; khong chan vong lap camera."""
        unique_phrases = list(dict.fromkeys(text.strip() for text in phrases if text.strip()))
        if not unique_phrases:
            return

        def warm() -> None:
            for text in unique_phrases:
                self._synthesize(text)

        threading.Thread(target=warm, name="voice-cache-prewarm", daemon=True).start()

    def synthesize(self, text: str) -> Path | None:
        """Return a cached speech file, creating it when necessary."""
        return self._synthesize(text)

    # -- worker --
    def _drain(self) -> None:
        while True:
            text, queued_at = self._queue.get()
            try:
                path = self._synthesize(text)
                if path is not None and time.monotonic() - queued_at <= self.max_queue_age_s:
                    if self.output is not None:
                        self.output(path)
                    else:
                        self._play(path)
            except Exception as error:  # noqa: BLE001 - voice khong duoc lam chet pipeline
                print(f"[Voice] Bỏ lời chào: {error}")
            finally:
                self._queue.task_done()

    def _synthesize(self, text: str) -> Path | None:
        pregen = self.phrase_files.get(text)
        if pregen is not None:
            path = Path(pregen)
            if path.is_file() and path.stat().st_size > 0:
                return path
        if self._tts_failed:
            return None
        cache = Path(self.cache_dir)
        try:
            cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        path = cache / _phrase_key(text, self.voice)
        if path.is_file() and path.stat().st_size > 0:
            return path
        try:
            asyncio.run(self._edge_save(text, path))
        except Exception:  # noqa: BLE001 - mat mang / API loi
            self._tts_failed = True
            return None
        return path if path.is_file() else None

    async def _edge_save(self, text: str, path: Path) -> None:
        import edge_tts

        await edge_tts.Communicate(text, voice=self.voice).save(str(path))

    @staticmethod
    def _play(path: Path) -> None:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
            check=False,
        )


__all__ = ["VoiceGreeter"]
