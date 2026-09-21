"""Fast, non-blocking ZeroTTS -> IMOU VisualTalk camera speaker.

Design goals
------------
* Keep ZeroTTS loaded once for the whole process.
* Never block video / face inference.
* Cache PCM on disk so repeated greetings start almost immediately.
* Prioritise gestures/arrivals over reminders.
* Drop stale queued speech instead of speaking old events later.
* Speak only through the IMOU camera (VisualTalk :8086), never laptop audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import os
from pathlib import Path
from queue import Empty, Full, PriorityQueue, Queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Iterable

MODEL_ID = "zeroweight-ai/ZeroTTS"
DEFAULT_VOICE = "hamy"
PERSONA_NAME = "Bé Xinh"
PCM_SAMPLE_RATE = 16_000
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * 2  # mono signed 16-bit
PCM_CHUNK_BYTES = 1280  # 40 ms at 16 kHz mono s16le; smoother under tracking load

# Lower number = higher priority.
PRIORITY_GESTURE = 0
PRIORITY_ARRIVAL = 10
PRIORITY_APPROACH = 20
PRIORITY_STAND = 30
PRIORITY_REMINDER = 50
PRIORITY_PREWARM = 90

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / "output" / "hamy_audio_cache"


@dataclass(order=True, frozen=True)
class SpeechRequest:
    priority: int
    sequence: int
    text: str = field(compare=False)
    created_s: float = field(compare=False)
    expires_s: float | None = field(compare=False, default=20.0)
    cache_only: bool = field(compare=False, default=False)
    fallback_text: str = field(compare=False, default="")


class CameraCheckInAnnouncer:
    """Long-lived ZeroTTS worker and sole owner of camera talk transport."""

    def __init__(
        self,
        *,
        host: str,
        username: str,
        password: str,
        helper_path: Path,
        port: int = 8086,
        voice: str = DEFAULT_VOICE,
        gain: float = 1.0,
        cooldown_s: float = 30.0,
        ffmpeg_bin: str | None = None,
        cache_dir: Path | None = None,
        runtime_synthesis: bool = False,
    ) -> None:
        self.host = host
        self.username = username
        self.password = password
        self.helper_path = helper_path
        self.port = int(port)
        self.voice = self._normalize_voice(voice)
        self.gain = max(0.1, min(3.0, float(gain)))
        self.cooldown_s = max(1.0, float(cooldown_s))
        self.ffmpeg_bin = ffmpeg_bin or self._resolve_ffmpeg()
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_synthesis = bool(runtime_synthesis)

        self._queue: PriorityQueue[SpeechRequest | tuple] = PriorityQueue(maxsize=256)
        self._sequence = itertools.count()
        self._last_by_key: dict[str, float] = {}
        self._prewarm_pending: set[str] = set()
        self._closed = threading.Event()
        self._muted = threading.Event()
        self._ready = threading.Event()
        self._speaking = threading.Event()
        self._working = threading.Event()
        self._talk_process_lock = threading.Lock()
        self._talk_process: subprocess.Popen | None = None
        self._last_line = ""
        self._last_spoken_monotonic = float("-inf")
        self._thread = threading.Thread(
            target=self._run,
            name="hamy-camera-speaker",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def from_env(cls) -> "CameraCheckInAnnouncer | None":
        """Create Bé Xinh whenever IMOU credentials + helper are available.

        CAMERA_TTS_ENABLED is intentionally ignored. Camera AIM treats Bé Xinh
        as an always-on subsystem; stopping the Camera AIM worker stops her.
        """
        host = os.getenv("IMOU_IP", "").strip()
        username = os.getenv("IMOU_USER", "").strip()
        password = os.getenv("IMOU_PASSWORD", "")
        if not host or not username or not password:
            print("[Bé Xinh] unavailable: missing IMOU_IP/USER/PASSWORD.")
            return None

        helper_raw = os.getenv("IMOU_TALK_HELPER", "").strip()
        if helper_raw:
            helper_path = Path(helper_raw)
        elif os.name == "nt":
            helper_path = Path(
                r"D:\imou-life-talk\deploy\dockge\imou-bridge\frigate_imou_talk_exec.py"
            )
        else:
            helper_path = Path("frigate_imou_talk_exec.py")

        if not helper_path.is_file():
            print(f"[Bé Xinh] unavailable: IMOU talk helper not found: {helper_path}")
            return None

        try:
            port = int(os.getenv("IMOU_TALK_PORT", "8086"))
        except ValueError:
            port = 8086
        try:
            gain = float(os.getenv("CAMERA_TTS_GAIN", "1.0"))
        except ValueError:
            gain = 1.0
        try:
            cooldown = float(os.getenv("CAMERA_TTS_COOLDOWN_SECONDS", "30"))
        except ValueError:
            cooldown = 30.0

        cache_raw = os.getenv("HAMY_AUDIO_CACHE_DIR", "").strip()
        cache_dir = Path(cache_raw) if cache_raw else _DEFAULT_CACHE_DIR

        return cls(
            host=host,
            username=username,
            password=password,
            helper_path=helper_path,
            port=port,
            voice=os.getenv("CAMERA_TTS_VOICE", DEFAULT_VOICE),
            gain=gain,
            cooldown_s=cooldown,
            ffmpeg_bin=os.getenv("FFMPEG_BIN", "").strip() or None,
            cache_dir=cache_dir,
            runtime_synthesis=os.getenv("HAMY_RUNTIME_SYNTHESIS", "false").strip().lower() in {"1", "true", "yes", "on"},
        )

    @property
    def enabled(self) -> bool:
        return (
            self._thread.is_alive()
            and not self._closed.is_set()
            and not self._muted.is_set()
        )

    @property
    def muted(self) -> bool:
        return self._muted.is_set()

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self.enabled

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "muted": self.muted,
            "ready": self.ready,
            "speaking": self.speaking,
            "working": self._working.is_set(),
            "queue_depth": self._queue.qsize(),
            "last_line": self._last_line,
            "runtime_synthesis": self.runtime_synthesis,
        }

    def set_muted(self, muted: bool) -> dict:
        """Mute/unmute immediately without stopping tracking or the worker."""
        if muted:
            self._muted.set()
            self._discard_live_requests()
            with self._talk_process_lock:
                process = self._talk_process
            if process is not None and process.poll() is None:
                process.kill()
            self._speaking.clear()
            print("[Bé Xinh] muted from dashboard.")
        else:
            self._muted.clear()
            print("[Bé Xinh] enabled from dashboard.")
        return self.status()

    def _discard_live_requests(self) -> None:
        keep: list[SpeechRequest] = []
        while True:
            try:
                request = self._queue.get_nowait()
            except Empty:
                break
            if isinstance(request, SpeechRequest) and request.cache_only:
                keep.append(request)
        for request in keep:
            try:
                self._queue.put_nowait(request)
            except Full:
                break

    def is_cached(self, text: str) -> bool:
        text = self._clean_text(text)
        return bool(text and self._cache_path(text).is_file())

    def prewarm(self, texts: Iterable[str], *, priority: int = PRIORITY_PREWARM) -> int:
        """Generate PCM cache in the background without playing it.

        Cached greetings survive process restarts, so after the first warmup
        arrival playback no longer waits for ZeroTTS synthesis.
        """
        added = 0
        for raw in texts:
            text = self._clean_text(raw)
            if not text or self.is_cached(text) or text in self._prewarm_pending:
                continue
            request = SpeechRequest(
                priority=int(priority),
                sequence=next(self._sequence),
                text=text,
                created_s=time.monotonic(),
                expires_s=None,
                cache_only=True,
            )
            try:
                self._queue.put_nowait(request)
            except Full:
                break
            self._prewarm_pending.add(text)
            added += 1
        return added

    def announce(self, person_id: str, display_name: str) -> bool:
        """Backward-compatible short arrival greeting API."""
        display_name = self._clean_text(display_name)
        person_id = str(person_id or "").strip()
        if not display_name or not person_id:
            return False
        return self.say(
            f"{display_name} tới rồi nè! Bé Xinh chào nha.",
            key=f"legacy-arrival:{person_id}",
            cooldown_s=self.cooldown_s,
            priority=PRIORITY_ARRIVAL,
            expires_s=12.0,
            fallback_text="Chào nha! Bé Xinh thấy bạn rồi.",
        )

    def say(
        self,
        text: str,
        *,
        key: str | None = None,
        cooldown_s: float = 0.0,
        priority: int = 50,
        expires_s: float | None = 20.0,
        fallback_text: str | None = None,
    ) -> bool:
        """Queue speech without blocking inference.

        If the requested personalised sentence is not cached but a short
        fallback is already cached, the fallback is used immediately. This is
        what prevents the user reaching their desk before the greeting starts.
        """
        text = self._clean_text(text)
        if not text or self._closed.is_set() or self._muted.is_set():
            return False

        now = time.monotonic()
        if key:
            last = self._last_by_key.get(key, float("-inf"))
            if now - last < max(0.0, cooldown_s):
                return False

        fallback = self._clean_text(fallback_text or "")
        selected = text
        if fallback and not self.is_cached(text) and self.is_cached(fallback):
            selected = fallback

        request = SpeechRequest(
            priority=int(priority),
            sequence=next(self._sequence),
            text=selected,
            created_s=now,
            expires_s=None if expires_s is None else max(0.5, float(expires_s)),
            cache_only=False,
            fallback_text=fallback,
        )
        try:
            self._queue.put_nowait(request)
        except Full:
            print("[Bé Xinh] speech queue full; dropped a stale/non-critical event.")
            return False

        if key:
            self._last_by_key[key] = now
        return True

    def wait_idle(self, timeout_s: float = 600.0) -> bool:
        """Wait until queued synthesis/cache work finishes (setup scripts only)."""
        deadline = time.monotonic() + max(0.1, timeout_s)
        idle_since: float | None = None
        while time.monotonic() < deadline:
            now = time.monotonic()
            idle = self._queue.qsize() == 0 and not self._working.is_set()
            if idle:
                if idle_since is None:
                    idle_since = now
                elif now - idle_since >= 0.30:
                    return True
            else:
                idle_since = None
            time.sleep(0.05)
        return False

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._queue.put_nowait(
                SpeechRequest(
                    priority=10_000,
                    sequence=next(self._sequence),
                    text="__HAMY_STOP__",
                    created_s=time.monotonic(),
                    expires_s=None,
                    cache_only=True,
                )
            )
        except Full:
            pass
        self._thread.join(timeout=10.0)

    def _run(self) -> None:
        # Production runtime is cache-first. Do NOT load ZeroTTS here: YOLO +
        # InsightFace are already using CUDA and loading another large model at
        # this point can make the Windows process terminate inside native CUDA
        # code without a Python traceback. ZeroTTS is loaded lazily only by the
        # offline prewarm path, or when HAMY_RUNTIME_SYNTHESIS=true explicitly.
        tts = None
        self._ready.set()
        mode = "cache-first" if not self.runtime_synthesis else "cache-first + lazy synthesis"
        print(f"[Bé Xinh] Ready. voice={self.voice} | {mode}")

        while not self._closed.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue

            request = item
            if request.text == "__HAMY_STOP__":
                return
            if self._muted.is_set() and not request.cache_only:
                continue

            if (
                request.expires_s is not None
                and time.monotonic() - request.created_s > request.expires_s
            ):
                continue

            self._working.set()
            try:
                selected_text = request.text
                pcm = self._read_cached_pcm(selected_text)

                # Prefer an already-cached fallback for real-time events.
                if not pcm and request.fallback_text:
                    fallback_pcm = self._read_cached_pcm(request.fallback_text)
                    if fallback_pcm:
                        selected_text = request.fallback_text
                        pcm = fallback_pcm

                # Offline prewarm is allowed to synthesize. Live runtime only
                # synthesizes on a miss when explicitly enabled.
                may_synthesize = request.cache_only or self.runtime_synthesis
                if not pcm and may_synthesize:
                    if tts is None:
                        try:
                            from zerotts import ZeroTTS
                            print("[Bé Xinh] Loading ZeroTTS model (cache build)...")
                            tts = ZeroTTS.from_pretrained(MODEL_ID)
                            print(f"[Bé Xinh] ZeroTTS cache builder ready. voice={self.voice}")
                        except Exception as error:  # noqa: BLE001
                            print(f"[Bé Xinh] ZeroTTS load failed: {error}")
                            tts = None
                    if tts is not None:
                        pcm = self._build_and_cache_pcm(tts, selected_text)

                self._prewarm_pending.discard(request.text)
                if request.cache_only:
                    continue

                if self._muted.is_set():
                    continue

                if not pcm:
                    print(
                        f"[Bé Xinh] cache miss; skipped live synthesis: {selected_text}"
                    )
                    continue

                print(f"[Bé Xinh] {selected_text}")
                self._speaking.set()
                try:
                    self._play_pcm(pcm)
                finally:
                    self._speaking.clear()
                self._last_line = selected_text
                self._last_spoken_monotonic = time.monotonic()
                print("[Bé Xinh] announced")
            except Exception as error:  # noqa: BLE001
                self._prewarm_pending.discard(request.text)
                print(f"[Bé Xinh] speak failed: {error}")
            finally:
                self._working.clear()

    def _read_cached_pcm(self, text: str) -> bytes:
        cache_path = self._cache_path(text)
        try:
            pcm = cache_path.read_bytes()
            if pcm:
                return pcm
        except OSError:
            pass
        return b""

    def _build_and_cache_pcm(self, tts, text: str) -> bytes:
        pcm = self._read_cached_pcm(text)
        if pcm:
            return pcm

        cache_path = self._cache_path(text)
        with tempfile.TemporaryDirectory(prefix="camera_aim_be_xinh_") as temp_dir:
            wav_path = Path(temp_dir) / "be_xinh.wav"
            audio = tts.synthesize(text, voice=self.voice)
            tts.save_audio(audio, str(wav_path))
            pcm = self._wav_to_pcm(wav_path)

        if pcm:
            temp_cache = cache_path.with_suffix(".tmp")
            try:
                temp_cache.write_bytes(pcm)
                temp_cache.replace(cache_path)
            except OSError:
                try:
                    temp_cache.unlink(missing_ok=True)
                except OSError:
                    pass
        return pcm

    def _cache_path(self, text: str) -> Path:
        payload = f"be-xinh-pcm-v4-cute\0{self.voice}\0{text}".encode("utf-8")
        digest = hashlib.sha1(payload).hexdigest()  # noqa: S324 - cache key only
        return self.cache_dir / f"{digest}.s16le"

    def _wav_to_pcm(self, wav_path: Path) -> bytes:
        # Short buffer + moderate gain. The limiter keeps the louder signal from
        # clipping the small IMOU speaker.
        # Cute/clear profile for the small IMOU speaker:
        # - short lead-in to protect the first syllable without delaying greeting too much;
        # - remove low rumble + harsh top end;
        # - slight 2.5 kHz presence lift for Vietnamese consonant clarity;
        # - 0.97x tempo makes the voice a touch softer/easier to follow;
        # - moderate gain + limiter avoids clipping and crackle.
        filters = (
            "adelay=180,"
            "highpass=f=110,"
            "lowpass=f=6200,"
            "equalizer=f=2500:t=q:w=1.2:g=1.5,"
            "atempo=0.97,"
            "volume=1.10,"
            "alimiter=limit=0.90,"
            "apad=pad_dur=0.65"
        )
        command = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-af",
            filters,
            "-ac",
            "1",
            "-ar",
            str(PCM_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            "pipe:1",
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg PCM conversion failed: {detail}")
        return bytes(result.stdout)

    def _play_pcm(self, pcm: bytes) -> None:
        if self._muted.is_set():
            return
        last_error: Exception | None = None
        for attempt in range(1, 3):
            try:
                self._play_pcm_once(pcm)
                return
            except Exception as error:  # noqa: BLE001
                last_error = error
                if self._muted.is_set():
                    return
                if attempt < 2:
                    print(f"[Bé Xinh] VisualTalk retry {attempt}/2: {error}")
                    # IMOU occasionally returns one transient 503 while its
                    # talk channel is being released. A long pause makes a
                    # greeting feel unrelated to the gesture; retry promptly.
                    time.sleep(0.40)
        raise RuntimeError(f"VisualTalk failed after 2 attempts: {last_error}")

    def _play_pcm_once(self, pcm: bytes) -> None:
        command = [
            sys.executable,
            "-u",
            str(self.helper_path),
            "--direct",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--serial",
            "CAMERA_AIM_HAMY",
            "--username",
            self.username,
            "--password",
            self.password,
            "--channel",
            "1",
            "--subtype",
            "0",
            "--type",
            "0",
            "--startup-delay",
            "0",
            "--input-codec",
            "s16le",
            "--input-sample-rate",
            str(PCM_SAMPLE_RATE),
            "--output-codec",
            "aac-adts",
            "--sample-rate",
            str(PCM_SAMPLE_RATE),
            "--aac-bitrate",
            "48000",
            "--volume-gain",
            str(self.gain),
            "--debug",
        ]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        with self._talk_process_lock:
            self._talk_process = process
        if process.stdin is None or process.stdout is None:
            process.kill()
            raise RuntimeError("VisualTalk helper pipes unavailable")

        line_queue: Queue[str | None] = Queue()

        def read_output() -> None:
            assert process.stdout is not None
            try:
                while True:
                    raw = process.stdout.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if line:
                        print(f"[IMOU] {line}")
                    line_queue.put(line)
            finally:
                line_queue.put(None)

        reader = threading.Thread(
            target=read_output,
            name="imou-talk-output",
            daemon=True,
        )
        reader.start()

        try:
            ready = False
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"VisualTalk helper exited early ({process.returncode})"
                    )
                try:
                    line = line_queue.get(timeout=0.20)
                except Empty:
                    continue
                if line is None:
                    break
                if "talk started" in line.lower():
                    ready = True
                    break

            if not ready:
                raise RuntimeError("VisualTalk did not report 'talk started'")

            # Cached PCM already starts with ~180 ms of silence, so another
            # 250 ms wait here only adds avoidable response latency. A short
            # guard is enough after the helper reports that VisualTalk started.
            time.sleep(0.08)

            # Pace against an absolute clock instead of adding one sleep per
            # chunk. This prevents scheduler jitter/drift from turning into
            # audible gaps when tracking is busy. 40 ms chunks also halve the
            # wake-up rate versus the old 20 ms loop.
            offset = 0
            sent = 0
            stream_started = time.perf_counter()
            while offset < len(pcm):
                chunk = pcm[offset : offset + PCM_CHUNK_BYTES]
                process.stdin.write(chunk)
                offset += len(chunk)
                sent += len(chunk)

                target = stream_started + (sent / PCM_BYTES_PER_SECOND)
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

            process.stdin.close()
            try:
                return_code = process.wait(timeout=12.0)
            except subprocess.TimeoutExpired as error:
                process.kill()
                raise RuntimeError("VisualTalk helper did not stop after PCM EOF") from error

            if return_code != 0:
                raise RuntimeError(f"VisualTalk helper exited with {return_code}")
        finally:
            try:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
            except Exception:
                pass
            if process.poll() is None:
                process.kill()
            reader.join(timeout=1.0)
            with self._talk_process_lock:
                if self._talk_process is process:
                    self._talk_process = None

    @staticmethod
    def _clean_text(value: str | None) -> str:
        return " ".join(str(value or "").split())

    @staticmethod
    def _normalize_voice(value: str | None) -> str:
        voice = (value or DEFAULT_VOICE).strip()
        if not voice or voice.lower().startswith("vi-vn-"):
            return DEFAULT_VOICE
        return voice

    @staticmethod
    def _resolve_ffmpeg() -> str:
        found = shutil.which("ffmpeg")
        if found:
            return found

        known = Path(
            r"C:\Users\PC\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe"
        )
        if known.is_file():
            return str(known)

        raise RuntimeError("ffmpeg not found. Set FFMPEG_BIN or add ffmpeg to PATH.")


__all__ = [
    "CameraCheckInAnnouncer",
    "PRIORITY_GESTURE",
    "PRIORITY_ARRIVAL",
    "PRIORITY_APPROACH",
    "PRIORITY_STAND",
    "PRIORITY_REMINDER",
    "PRIORITY_PREWARM",
]
