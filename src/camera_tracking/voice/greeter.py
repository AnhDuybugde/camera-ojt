"""Chao bang giong noi (khong block inference).

- Hang doi uu tien + thread rieng phat am thanh; uu tien WAV tao san
  (scripts/build_greeting_wavs.py, 1 cau = 1 file -> phat 1 session P2P
  khong ngat quang), fallback TTS edge-tts vi-VN cache mp3 theo cau.
- Phat bang ffplay (di kem ffmpeg, khong them dep) khi backend local.
- Hai loai yeu cau:
  + Chu dong (vay tay / noi hello): loa ranh thi phat NGAY, khong cooldown;
    chong spam bang "khoang lang" 5s/nguoi sau moi lan phat + gop queue
    theo nguoi (dang phat thi nguoi khac xep cho, cung nguoi thi bo qua).
  + Tu dong (thay mat o cua): moi nguoi cach nhau 10s.
- Moi cau chao la 1 ban thu hoan chinh, khong ghep noi 2 doan am thanh.
"""
from __future__ import annotations

import asyncio
import hashlib
import heapq
import itertools
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
    # Khoang lang chong spam sau moi lan phat cho 1 nguoi (ap dung ca hai
    # loai): trong khoang nay yeu cau chu dong cua chinh nguoi do bi bo qua.
    proactive_quiet_s: float = 5.0
    output: Callable[..., None] | None = None
    # Goi sau khi backend phat xong, dung de inhibit mic camera khoi nghe
    # lai chinh cau chao vua phat.
    on_spoken: Callable[[str], None] | None = None
    # WAV tao san (ZeroTTS): {cau chao dung nguyen van: duong dan file}.
    # Khop thi dung ngay, khong goi edge-tts (offline hoan toan).
    phrase_files: dict[str, str | Path] = field(default_factory=dict)
    _last_spoken: dict[tuple, float] = field(
        default_factory=dict, init=False)
    # Hang doi uu tien (chu dong truoc, tu dong sau; cung muc thi den
    # truoc phat truoc). Moi nguoi chi co toi da 1 luot cho (gop theo nguoi).
    _queue: queue.PriorityQueue = field(
        default_factory=queue.PriorityQueue, init=False)
    _seq: itertools.count = field(default_factory=itertools.count, init=False)
    # Luot cho theo nguoi: identity -> (seq, khoa dat cho, gia tri dat cho).
    # Khoa dat cho None = luot chu dong (khong giu cooldown).
    _pending: dict[str, tuple[int, tuple | None, float | None]] = field(
        default_factory=dict, init=False)
    _cancelled: set[int] = field(default_factory=set, init=False)
    _inflight: set[str] = field(default_factory=set, init=False)
    _quiet_until: dict[str, float] = field(default_factory=dict, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _tts_failed: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._drain, name="voice-greeter", daemon=True)
        self._thread.start()

    def _cooldown_key(
        self,
        day: str,
        person_id: str | None,
        global_id: int | None,
        channel: int | None,
    ) -> tuple:
        """Khoa cooldown theo loa: cung nguoi nhung khac loa van chao rieng.

        channel=None giu khoa cu (day, identity) de tuong thich nguoc.
        """
        identity_key = person_id or f"unknown:{global_id if global_id is not None else 'any'}"
        if channel is None:
            return (day, identity_key)
        return (day, identity_key, int(channel))

    @staticmethod
    def _identity_key(
        person_id: str | None, global_id: int | None
    ) -> str:
        return person_id or f"unknown:{global_id if global_id is not None else 'any'}"

    def wave_greet(
        self,
        *,
        day: str,
        person_id: str | None,
        display_name: str | None,
        now_s: float,
        global_id: int | None = None,
        cooldown_s: float | None = None,
        channel: int | None = None,
        proactive: bool = False,
    ) -> bool:
        """Xep 1 loi chao. Tra True khi da xep (loa ranh thi phat ngay).

        - Chu dong (proactive=True, vay tay / noi hello): KHONG cooldown.
          Cung nguoi dang phat hoac moi phat trong khoang lang thi bo qua
          (chong spam); nguoi khac dang phat thi xep cho (gop 1 luot/nguoi).
        - Tu dong (proactive=False, thay mat): moi nguoi cach nhau
          cooldown (mac dinh 10s, la hay quen nhu nhau).
        - Dat cho truoc (reservation) de 2 trigger lien tiep khong xen nhau.
          Neu am thanh that su khong phat duoc (het TTL / loi TTS / loi P2P),
          worker se hoan reservation do.

        channel = kenh loa (1=phong A, 2=cua B). Khac loa thi cooldown
        rieng (nguoi di tu phong ra cua van duoc chao o cua).
        """
        identity_key = self._identity_key(person_id, global_id)
        out_channel = None if channel is None else int(channel)
        if person_id and display_name:
            text = f"Xin chào {display_name}"
        else:
            text = self.unknown_phrase
        with self._lock:
            if proactive:
                # Dang phat cho chinh nguoi nay, hoac moi phat xong trong
                # khoang lang -> bo qua (vay lien tuc khong spam).
                if identity_key in self._inflight:
                    return False
                if now_s < self._quiet_until.get(identity_key, float("-inf")):
                    return False
                # Da co luot cho (tu dong hoac chu dong) -> huy luot cu,
                # nang len chu dong phat truoc, khong xep trung.
                old = self._pending.pop(identity_key, None)
                if old is not None:
                    old_seq, old_key, old_value = old
                    self._cancelled.add(old_seq)
                    if old_key is not None and (
                        self._last_spoken.get(old_key) == old_value
                    ):
                        self._last_spoken.pop(old_key, None)
                seq = next(self._seq)
                self._pending[identity_key] = (seq, None, None)
                self._queue.put((
                    0, seq,
                    (day, identity_key, text, time.monotonic(), now_s,
                     out_channel, "proactive", seq),
                ))
                return True
            key = self._cooldown_key(day, person_id, global_id, channel)
            # Nguoi la dung cooldown ngan hon de palm chao ngay lap tuc.
            if cooldown_s is None:
                cooldown_s = self.cooldown_s if person_id else self.unknown_cooldown_s
            if now_s - self._last_spoken.get(key, float("-inf")) < cooldown_s:
                return False
            # Da co luot cho thi thoi (gop theo nguoi).
            if identity_key in self._pending:
                return False
            self._last_spoken[key] = now_s
            seq = next(self._seq)
            self._pending[identity_key] = (seq, key, now_s)
            self._queue.put((
                1, seq,
                (day, identity_key, text, time.monotonic(), now_s,
                 out_channel, "auto", seq),
            ))
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
        channel: int | None = None,
        proactive: bool = False,
    ) -> bool:
        """Chao khi nhan dien mat (dung truoc camera), khong can vay tay.

        Mac dinh la tu dong (cach nhau cooldown). Gọi voi proactive=True
        khi day la phan hoi cho yeu cau chu dong (vay tay / voice).
        """
        return self.wave_greet(
            day=day,
            person_id=person_id,
            display_name=display_name,
            now_s=now_s,
            global_id=global_id,
            cooldown_s=cooldown_s,
            channel=channel,
            proactive=proactive,
        )

    def cancel_pending_unknown(
        self, *, global_id: int | None
    ) -> bool:
        """Huy luot cho "quy khach" cua 1 GID (mat vua xac nhan ten).

        Tra True khi co luot cho bi huy. Dung truoc khi xep loi chao ten
        de nguoi vua dinh danh chi nghe 1 cau chao ten, khong kem cau
        "quy khach" da xep truoc do.
        """
        identity_key = f"unknown:{global_id}"
        with self._lock:
            old = self._pending.pop(identity_key, None)
            if old is None:
                return False
            old_seq, old_key, old_value = old
            self._cancelled.add(old_seq)
            if old_key is not None and (
                self._last_spoken.get(old_key) == old_value
            ):
                self._last_spoken.pop(old_key, None)
            return True

    def unknown_spoken_age(
        self,
        *,
        day: str,
        global_id: int | None,
        channel: int | None,
        now_s: float,
    ) -> float | None:
        """So giay tu luc "quy khach" cua GID duoc phat xong.

        None = chua phat (hoac reservation da bi huy/rollback). Dung de
        bo loi chao ten khi cau "quy khach" vua phat (tranh noi 2 cau).
        """
        identity_key = f"unknown:{global_id}"
        keys = [(day, identity_key)]
        if channel is not None:
            keys.append((day, identity_key, int(channel)))
        best: float | None = None
        with self._lock:
            for key in keys:
                at = self._last_spoken.get(key)
                if at is None:
                    continue
                try:
                    age = float(now_s) - float(at)
                except (TypeError, ValueError):
                    continue
                if age >= 0 and (best is None or age < best):
                    best = age
        return best

    def cooldown_remaining(
        self,
        *,
        day: str,
        person_id: str | None,
        now_s: float,
        global_id: int | None = None,
        channel: int | None = None,
    ) -> float:
        """So giay con lai truoc khi danh tinh nay duoc chao tiep (0 = san sang)."""
        limit = self.cooldown_s if person_id else self.unknown_cooldown_s
        with self._lock:
            last = self._last_spoken.get(
                self._cooldown_key(day, person_id, global_id, channel),
                float("-inf"),
            )
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
    def _rollback(
        self,
        day: str,
        identity_key: str,
        enqueued_now_s: float,
        channel: int | None = None,
    ) -> None:
        """Hoan reservation khi am thanh khong phat duoc (het TTL/loi).

        Chi xoa khi gia tri hien tai van la lan dat cho nay (tranh xoa
        reservation moi hon cua 1 loi chao thanh cong sau do).
        """
        key = (day, identity_key) if channel is None else (day, identity_key, channel)
        with self._lock:
            if self._last_spoken.get(key) == enqueued_now_s:
                self._last_spoken.pop(key, None)

    def _output_takes_channel(self) -> bool:
        """Output co nhan tham so kenh loa thu 2 khong (cache lan dau).

        Output cu 1 tham so (path) van chay: bo qua kenh, phat loa mac dinh.
        """
        cached = getattr(self, "_takes_channel", None)
        if cached is not None:
            return bool(cached)
        takes = False
        if self.output is not None:
            try:
                import inspect

                params = list(inspect.signature(self.output).parameters.values())
                kinds = (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                positional = sum(1 for p in params if p.kind in kinds)
                variadic = any(
                    p.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )
                    for p in params
                )
                takes = positional >= 2 or variadic
            except (TypeError, ValueError):
                takes = False
        self._takes_channel: bool | None = takes
        return takes

    def _drain(self) -> None:
        while True:
            wrapped = self._queue.get()
            try:
                # Item moi (uu tien, seq, payload); tuong thich item cu
                # 6-tuple (co kenh), 5-tuple va 2-tuple (text, queued_at).
                if len(wrapped) == 3:
                    _prio, seq, item = wrapped
                    if seq in self._cancelled:
                        self._cancelled.discard(seq)
                        continue
                else:
                    seq, item = -1, wrapped
                if len(item) == 8:
                    (day, identity_key, text, queued_at,
                     enqueued_now_s, out_channel, _kind, _iseq) = item
                elif len(item) == 6:
                    (day, identity_key, text, queued_at,
                     enqueued_now_s, out_channel) = item
                elif len(item) == 5:
                    day, identity_key, text, queued_at, enqueued_now_s = item
                    out_channel = None
                else:
                    text, queued_at = item
                    day, identity_key, enqueued_now_s = "", "", -1.0
                    out_channel = None
                with self._lock:
                    pending = self._pending.get(identity_key)
                    if pending is not None and pending[0] == seq:
                        self._pending.pop(identity_key, None)
                    self._inflight.add(identity_key)
                path = self._synthesize(text)
                if path is None:
                    with self._lock:
                        self._inflight.discard(identity_key)
                    self._rollback(day, identity_key, enqueued_now_s, out_channel)
                    continue
                if time.monotonic() - queued_at > self.max_queue_age_s:
                    # Qua cu: bo de khoi phat tre, dong thoi tra cooldown
                    # de lan ke tiep duoc chao ngay.
                    with self._lock:
                        self._inflight.discard(identity_key)
                    self._rollback(day, identity_key, enqueued_now_s, out_channel)
                    continue
                try:
                    if self.output is not None:
                        if out_channel is None or not self._output_takes_channel():
                            self.output(path)
                        else:
                            self.output(path, out_channel)
                    else:
                        self._play(path)
                    if self.on_spoken is not None:
                        try:
                            self.on_spoken(text)
                        except Exception as callback_error:  # noqa: BLE001
                            print(f"[Voice] callback sau phát lỗi: {callback_error}")
                    # Chi bao thanh cong sau khi backend (P2P/local) da tra
                    # ve, khong nham voi log trigger moi chi xep hang.
                    if out_channel is None:
                        print(f"[Voice] Đã phát: {text}")
                    else:
                        print(f"[Voice] Đã phát (loa ch{out_channel}): {text}")
                    # Phat xong: mo khoang lang chong spam cho nguoi nay +
                    # nap lai moc cooldown tu dong tu thoi diem phat xong
                    # (gio pipeline ≈ gio xep + thoi gian doi/phat).
                    pipeline_end = enqueued_now_s + max(
                        0.0, time.monotonic() - queued_at)
                    with self._lock:
                        self._inflight.discard(identity_key)
                        self._quiet_until[identity_key] = (
                            pipeline_end + max(0.0, self.proactive_quiet_s)
                        )
                        self._last_spoken[(day, identity_key)] = pipeline_end
                        if out_channel is not None:
                            self._last_spoken[
                                (day, identity_key, out_channel)] = pipeline_end
                except Exception:
                    # P2P/loa loi: tra cooldown, main loop se log o lan toi.
                    with self._lock:
                        self._inflight.discard(identity_key)
                    self._rollback(day, identity_key, enqueued_now_s, out_channel)
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
