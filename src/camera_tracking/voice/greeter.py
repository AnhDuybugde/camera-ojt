"""Chao bang giong noi khi co nguoi gio tay 5 ngon (khong block inference).

- Hang doi + thread rieng phat am thanh; uu tien WAV tao san bang ZeroTTS
  (scripts/build_greeting_wavs.py, 1 cau = 1 file -> phat 1 session P2P
  khong ngat quang), fallback TTS edge-tts vi-VN cache mp3 theo cau.
- Phat bang ffplay (di kem ffmpeg, khong them dep) khi backend local.
- Thong nhat: la hay quen deu cooldown 10s, chi kich hoat khi gio tay.
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
    # Thong nhat 10s cho ca quen lan la, chi kich hoat khi gio tay 5 ngon.
    cooldown_s: float = 10.0
    # Cooldown rieng cho nguoi la (giu de tuy chinh, mac dinh cung 10s).
    unknown_cooldown_s: float = 10.0
    unknown_phrase: str = "Xin chào quý khách"
    max_queue_age_s: float = 5.0
    output: Callable[[Path], None] | None = None
    # Goi sau khi backend phat xong, dung de inhibit mic camera khoi nghe
    # lai chinh cau chao vua phat.
    on_spoken: Callable[[str], None] | None = None
    # WAV tao san (ZeroTTS): {cau chao dung nguyen van: duong dan file}.
    # Khop thi dung ngay, khong goi edge-tts (offline hoan toan).
    phrase_files: dict[str, str | Path] = field(default_factory=dict)
    _last_spoken: dict[tuple[str, str], float] = field(
        default_factory=dict, init=False)
    _queue: queue.Queue = field(default_factory=queue.Queue, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _tts_failed: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

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
        cooldown_s: float | None = None,
    ) -> bool:
        """Xep 1 loi chao neu qua cooldown. Tra True khi da xep.

        Dat cho truoc (reservation) de 2 trigger lien tiep khong xen nhau.
        Neu am thanh that su khong phat duoc (het TTL / loi TTS / loi P2P),
        worker se hoan reservation do de he thong luon o trang thai ready
        thay vi bi tru cooldown oan.
        """
        identity_key = person_id or f"unknown:{global_id if global_id is not None else 'any'}"
        key = (day, identity_key)
        # Nguoi la dung cooldown ngan hon de palm chao ngay lap tuc.
        if cooldown_s is None:
            cooldown_s = self.cooldown_s if person_id else self.unknown_cooldown_s
        with self._lock:
            if now_s - self._last_spoken.get(key, float("-inf")) < cooldown_s:
                return False
            self._last_spoken[key] = now_s
        if person_id and display_name:
            text = f"Xin chào {display_name}"
        else:
            text = self.unknown_phrase
        self._queue.put((day, identity_key, text, time.monotonic(), now_s))
        return True

    def face_greet(
        self,
        *,
        day: str,
        person_id: str | None,
        display_name: str | None,
        now_s: float,
        global_id: int | None = None,
        cooldown_s: float | None = None,
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
            cooldown_s=cooldown_s,
        )

    def cooldown_remaining(
        self,
        *,
        day: str,
        person_id: str | None,
        now_s: float,
        global_id: int | None = None,
    ) -> float:
        """So giay con lai truoc khi danh tinh nay duoc chao tiep (0 = san sang)."""
        identity_key = person_id or f"unknown:{global_id if global_id is not None else 'any'}"
        limit = self.cooldown_s if person_id else self.unknown_cooldown_s
        with self._lock:
            last = self._last_spoken.get((day, identity_key), float("-inf"))
        return max(0.0, limit - (now_s - last))

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
    def _rollback(self, day: str, identity_key: str, enqueued_now_s: float) -> None:
        """Hoan reservation khi am thanh khong phat duoc (het TTL/loi).

        Chi xoa khi gia tri hien tai van la lan dat cho nay (tranh xoa
        reservation moi hon cua 1 loi chao thanh cong sau do).
        """
        key = (day, identity_key)
        with self._lock:
            if self._last_spoken.get(key) == enqueued_now_s:
                self._last_spoken.pop(key, None)

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            try:
                # Tuong thich item cu 2-tuple (text, queued_at).
                if len(item) == 5:
                    day, identity_key, text, queued_at, enqueued_now_s = item
                else:
                    text, queued_at = item
                    day, identity_key, enqueued_now_s = "", "", -1.0
                path = self._synthesize(text)
                if path is None:
                    self._rollback(day, identity_key, enqueued_now_s)
                    continue
                if time.monotonic() - queued_at > self.max_queue_age_s:
                    # Qua cu (P2P dang ban): bo de khoi phat tre, dong thoi
                    # tra cooldown de lan gio tay ke tiep duoc chao ngay.
                    self._rollback(day, identity_key, enqueued_now_s)
                    continue
                try:
                    if self.output is not None:
                        self.output(path)
                    else:
                        self._play(path)
                    if self.on_spoken is not None:
                        try:
                            self.on_spoken(text)
                        except Exception as callback_error:  # noqa: BLE001
                            print(f"[Voice] callback sau phát lỗi: {callback_error}")
                    # Chi bao thanh cong sau khi backend (P2P/local) da tra
                    # ve, khong nham voi log trigger moi chi xep hang.
                    print(f"[Voice] Đã phát: {text}")
                except Exception:
                    # P2P/loa loi: tra cooldown, main loop se log o lan toi.
                    self._rollback(day, identity_key, enqueued_now_s)
                    raise
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
