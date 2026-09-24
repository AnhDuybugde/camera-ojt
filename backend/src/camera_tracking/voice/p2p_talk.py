"""Phat loa camera Imou qua P2P VisualTalk (thay backend imou_web).

Luong moi (vendor tu test-sound-camera-imou, goc tu imou-life research):

    TTS mp3 (VoiceGreeter / edge-tts, cache offline)
      -> ffmpeg: AAC-LC/ADTS, mono, 16 kHz
      -> DHAV interleaved audio frames
      -> visualtalk.xav talkback session
      -> DHP2P/PTCP relay tunnel (remote camera port 8086)
      -> loa camera

Khac backend cu (imou_web):
- Khong can Chrome/Edge, khong can virtual mic, khong mo mic laptop.
- Khong can IMOU_APP_ID / IMOU_APP_SECRET / kitToken / WebSDK / bridge 8767.
- Chi can serial camera + device password (safety code).
- Moi loi chao mo 1 tunnel P2P moi trong background thread cua VoiceGreeter
  nen khong block inference (giong backend cu).

Gesture giu nguyen: MediaPipe Hands + WaveDetector -> employee dung truoc
camera vay tay -> ``VoiceGreeter.wave_greet`` -> output nay phat
``Xin chào <ten>`` (quen) / ``Xin chào quý khách`` (la).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

try:
    from camera_tracking.voice.p2p.imou_dhp2p import DHP2PTunnel, p2p_handshake
    from camera_tracking.voice.p2p.imou_pure_talk import run_visualtalk
except ImportError:  # pragma: no cover - standalone script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent / "p2p"))
    from imou_dhp2p import DHP2PTunnel, p2p_handshake  # type: ignore[no-redef]
    from imou_pure_talk import run_visualtalk  # type: ignore[no-redef]


class P2PTalkError(RuntimeError):
    """Raised when the P2P talkback path cannot deliver audio."""


@dataclass(frozen=True, slots=True)
class ImouP2PCredentials:
    """Chi can serial + device password (safety code)."""

    serial: str
    password: str
    username: str = "admin"

    @classmethod
    def from_env(cls, username: str = "admin") -> ImouP2PCredentials:
        serial = os.getenv("IMOU_DEVICE_ID", "").strip().strip("'\"")
        # Thu tu uu tien: bien moi chuyen dung -> bien RTSP cu -> safety code cu.
        # Nhieu camera IMOU dan dung safety code lam luon RTSP password.
        password = ""
        for name in (
            "IMOU_CAMERA_PASSWORD",
            "IMOU_PASSWORD",
            "IMOU_DEVICE_CODE",
            "IMOU_DEVICE_PASSWORD",
        ):
            candidate = os.getenv(name, "").strip().strip("'\"")
            if candidate:
                password = candidate
                break
        missing = []
        if not serial:
            missing.append("IMOU_DEVICE_ID")
        if not password:
            missing.append("IMOU_CAMERA_PASSWORD (hoac IMOU_PASSWORD / IMOU_DEVICE_CODE)")
        if missing:
            raise P2PTalkError(f"Missing Imou P2P configuration: {', '.join(missing)}")
        return cls(serial=serial, password=password, username=username)


def find_ffmpeg() -> str:
    """Tim ffmpeg: uu tien imageio-ffmpeg bundle, fallback system ffmpeg."""
    try:
        import imageio_ffmpeg  # type: ignore

        exe = str(imageio_ffmpeg.get_ffmpeg_exe())
        if exe and Path(exe).is_file():
            return exe
    except (ImportError, RuntimeError):
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise P2PTalkError(
        "ffmpeg not found; chay: python -m pip install imageio-ffmpeg"
    )


def convert_file_to_aac_adts(
    input_path: str | Path,
    *,
    sample_rate: int = 16000,
    volume: float = 1.0,
) -> bytes:
    """Convert TTS mp3 / WAV bat ky sang AAC-LC/ADTS mono 16 kHz (bytes)."""
    src = Path(input_path)
    if not src.is_file():
        raise P2PTalkError(f"Speech audio does not exist: {src}")
    ffmpeg = find_ffmpeg()
    volume = max(0.0, min(1.0, float(volume)))
    env = dict(os.environ)
    # Bao dam ffmpeg con cua imageio-ffmpeg chay duoc khi goi bang ten.
    env["Path"] = str(Path(ffmpeg).parent) + ";" + env.get("Path", "")
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-filter:a",
            f"volume={volume:.3f}",
            "-ar",
            str(int(sample_rate)),
            "-ac",
            "1",
            "-c:a",
            "aac",
            "-profile:a",
            "aac_low",
            "-b:a",
            "32k",
            "-f",
            "adts",
            "-",
        ],
        check=False,
        capture_output=True,
        env=env,
    )
    if proc.returncode != 0 or not proc.stdout:
        detail = (proc.stderr or b"").decode("utf-8", "replace").strip()[:300]
        raise P2PTalkError(f"ffmpeg AAC convert failed: {detail or 'no output'}")
    return bytes(proc.stdout)


async def _send_aac(
    aac: bytes,
    creds: ImouP2PCredentials,
    *,
    channel: int = 1,
    timeout: float = 20.0,
    attempts: int = 3,
    retry_delay: float = 8.0,
    bind_host: str = "127.0.0.1",
    bind_port: int = 18086,
    sample_rate: int = 16000,
    debug: bool = False,
) -> int:
    """Mo relay tunnel tam thoi roi gui AAC qua visualtalk.xav. Tra so frames."""
    ptcp = await p2p_handshake(
        creds.serial,
        relay_mode=True,
        dtype=0,
        username=creds.username,
        password=creds.password,
        debug=debug,
    )
    tunnel = DHP2PTunnel(ptcp, 8086, debug=debug)
    server_task = asyncio.create_task(tunnel.start(bind_host, bind_port))
    await asyncio.sleep(2.0)
    try:
        args = SimpleNamespace(
            host=bind_host,
            port=bind_port,
            username=creds.username,
            channel=int(channel),
            subtype=0,
            encrypt=3,
            track1=0,
            track2=0,
            talk_track=64,
            media_track=5,
            sdp=None,
            open_only=False,
            dump_body=False,
            codec="aac-adts",
            sample_rate=int(sample_rate),
            frame_ms=64,
            timeout=float(timeout),
            attempts=int(attempts),
            retry_delay=float(retry_delay),
            debug=bool(debug),
        )
        # run_visualtalk dong bo (socket) -> chay trong thread de khoi chan loop.
        # No se raise RuntimeError neu camera tu choi (Cseq >= 400).
        def _run() -> int:
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                code = run_visualtalk(args, creds.password, aac)
            if code != 0:
                raise P2PTalkError(f"VisualTalk send failed (exit={code})")
            return code

        # run_visualtalk tu in "Cseq ..." ra stdout; giu lai de khong rac log
        # pipeline nhung van tra exit code. Muon debug thi bat debug=True o
        # config va xem exception.
        await asyncio.to_thread(_run)
        return 0
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


def send_audio_file(
    audio_path: str | Path,
    creds: ImouP2PCredentials | None = None,
    *,
    channel: int = 1,
    timeout: float = 20.0,
    attempts: int = 3,
    retry_delay: float = 8.0,
    sample_rate: int = 16000,
    volume: float = 1.0,
) -> None:
    """Convert + gui 1 file am thanh (mp3 TTS / wav) ra loa camera. Block."""
    creds = creds or ImouP2PCredentials.from_env()
    aac = convert_file_to_aac_adts(
        audio_path, sample_rate=sample_rate, volume=volume
    )
    try:
        asyncio.run(
            _send_aac(
                aac,
                creds,
                channel=channel,
                timeout=timeout,
                attempts=attempts,
                retry_delay=retry_delay,
                sample_rate=sample_rate,
            )
        )
    except P2PTalkError:
        raise
    except Exception as error:
        raise P2PTalkError(f"P2P talk failed: {error}") from error


class PersistentP2PTunnel:
    """Duong ham P2P giu san qua cac lan chao (1 tunnel cho ca 2 loa).

    Bat tay cloud + mo tunnel chi 1 lan; cac lan chao sau chi mo phien
    thoai + stream am thanh (~vai giay thay vi 8-20s). Tunnel co heartbeat
    nen tu giu; hong thi tu dung lai va bat tay moi o lan chao ke tiep.
    Kenh loa la tham so cua tung phien thoai, khong phai cua tunnel.
    """

    def __init__(
        self,
        creds: ImouP2PCredentials,
        *,
        bind_host: str = "127.0.0.1",
        bind_port: int = 18086,
        establish_timeout: float = 45.0,
        debug: bool = False,
    ) -> None:
        self.creds = creds
        self.bind_host = bind_host
        self.bind_port = int(bind_port)
        self.establish_timeout = float(establish_timeout)
        self.debug = debug
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server_task: asyncio.Task | None = None
        self._tunnel: DHP2PTunnel | None = None
        self._up = False

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None and self._thread is not None and self._thread.is_alive():
            return self._loop
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._thread = threading.Thread(
            target=loop.run_forever, name="p2p-tunnel", daemon=True)
        self._thread.start()
        return loop

    async def _establish(self) -> tuple[str, int]:
        if self._server_task is not None:
            self._server_task.cancel()
            try:
                await self._server_task
            except (asyncio.CancelledError, Exception):
                pass
            self._server_task = None
            await asyncio.sleep(0.2)
        ptcp = await p2p_handshake(
            self.creds.serial,
            relay_mode=True,
            dtype=0,
            username=self.creds.username,
            password=self.creds.password,
            debug=self.debug,
        )
        try:
            old = self._tunnel
            if old is not None:
                try:
                    old.ptcp.close()
                except Exception:
                    pass
            self._tunnel = DHP2PTunnel(ptcp, 8086, debug=self.debug)
            assert self._tunnel is not None
            self._server_task = asyncio.create_task(
                self._tunnel.start(self.bind_host, self.bind_port))
            await asyncio.sleep(1.0)
            self._up = True
            return self.bind_host, self.bind_port
        except Exception:
            try:
                ptcp.close()
            except Exception:
                pass
            self._up = False
            raise

    def ensure(self) -> tuple[str, int]:
        """Tra (host, port) san sang; bat tay moi neu tunnel sup."""
        if self._up and self._server_task is not None and not self._server_task.done():
            return self.bind_host, self.bind_port
        self._up = False
        loop = self._ensure_loop()
        try:
            future = asyncio.run_coroutine_threadsafe(self._establish(), loop)
            return future.result(timeout=self.establish_timeout)
        except Exception as error:
            self._up = False
            detail = str(error) or type(error).__name__
            raise P2PTalkError(
                f"P2P tunnel establish failed: {detail}") from error

    async def _reap(self, task: asyncio.Task) -> None:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    def invalidate(self) -> None:
        """Danh dau sup; lan chao ke tiep se bat tay lai."""
        self._up = False
        task, loop = self._server_task, self._loop
        self._server_task = None
        if task is not None and loop is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
                asyncio.run_coroutine_threadsafe(self._reap(task), loop)
            except RuntimeError:
                pass

    async def _shutdown(self) -> None:
        task, tunnel = self._server_task, self._tunnel
        self._server_task = None
        self._tunnel = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        # Xoi sach reader/heartbeat ma tunnel tu huy khong doi.
        for _ in range(20):
            pending = [
                t for t in asyncio.all_tasks()
                if t is not asyncio.current_task() and not t.done()
            ]
            if not pending:
                break
            for t in pending:
                t.cancel()
            await asyncio.sleep(0)
        if tunnel is not None:
            try:
                tunnel.ptcp.close()
            except Exception:
                pass

    def close(self) -> None:
        self._up = False
        loop, thread = self._loop, self._thread
        self._loop = None
        self._thread = None
        if loop is not None and thread is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
                future.result(timeout=5.0)
            except Exception:
                pass
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
            if thread is not threading.current_thread():
                thread.join(timeout=5.0)


def _visualtalk_once(
    aac: bytes,
    creds: ImouP2PCredentials,
    host: str,
    port: int,
    *,
    channel: int,
    timeout: float,
    sample_rate: int,
) -> None:
    """Mo 1 phien thoai + stream AAC (1 thu, khong tu retry)."""
    args = SimpleNamespace(
        host=host,
        port=port,
        username=creds.username,
        channel=int(channel),
        subtype=0,
        encrypt=3,
        track1=0,
        track2=0,
        talk_track=64,
        media_track=5,
        sdp=None,
        open_only=False,
        dump_body=False,
        codec="aac-adts",
        sample_rate=int(sample_rate),
        frame_ms=64,
        timeout=float(timeout),
        attempts=1,
        retry_delay=2.0,
        debug=False,
    )

    def _run() -> int:
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = run_visualtalk(args, creds.password, aac)
        if code != 0:
            raise P2PTalkError(f"VisualTalk send failed (exit={code})")
        return code

    try:
        _run()
    except P2PTalkError:
        raise
    except Exception as error:
        raise P2PTalkError(f"P2P talk failed: {error}") from error


class ImouP2PTalkOutput:
    """Callable adapter cho ``VoiceGreeter(output=...)`` (giong ImouAudioTalkOutput)."""

    def __init__(
        self,
        creds: ImouP2PCredentials | None = None,
        *,
        channel: int = 1,
        timeout: float = 20.0,
        attempts: int = 2,
        retry_delay: float = 5.0,
        sample_rate: int = 16000,
        volume: float = 1.0,
        persistent: bool = True,
        bind_host: str = "127.0.0.1",
        bind_port: int = 18086,
        establish_timeout: float = 45.0,
        retry_persistent: bool = True,
    ) -> None:
        self.creds = creds or ImouP2PCredentials.from_env()
        self.channel = int(channel)
        self.timeout = float(timeout)
        self.attempts = int(attempts)
        self.retry_delay = float(retry_delay)
        self.sample_rate = int(sample_rate)
        self.volume = max(0.0, min(1.0, float(volume)))
        self.persistent = bool(persistent)
        self.bind_host = str(bind_host)
        self.bind_port = int(bind_port)
        self.retry_persistent = bool(retry_persistent)
        self._tunnel = (
            PersistentP2PTunnel(
                self.creds,
                bind_host=self.bind_host,
                bind_port=self.bind_port,
                establish_timeout=float(establish_timeout),
            )
            if self.persistent
            else None
        )
        self._aac_cache: dict[tuple, bytes] = {}
        self._aac_lock = threading.Lock()

    def warmup(self, *, raise_on_error: bool = False) -> None:
        """Mo san tunnel o nen (lan chao dau khong mat tien bat tay).

        raise_on_error=True: nem P2PTalkError thay vi chi in log (dung khi
        caller muon cho den khi san sang that, vd tro ly Ni).
        """
        if self._tunnel is None:
            return
        try:
            self._tunnel.ensure()
            print("[Voice:P2P] Duong truyen loa da san sang.")
        except Exception as error:  # noqa: BLE001 - se thu lai o lan chao dau
            print(f"[Voice:P2P] Warmup chua xong ({error}); se mo o lan chao dau.")
            if raise_on_error:
                detail = str(error) or type(error).__name__
                raise P2PTalkError(f"P2P warmup failed: {detail}") from error

    def _convert_cached(self, audio_path: str | Path) -> bytes:
        src = Path(audio_path)
        try:
            stat = src.stat()
            key = (str(src), stat.st_size, stat.st_mtime_ns,
                   self.sample_rate, self.volume)
        except OSError:
            key = (str(src), -1, -1, self.sample_rate, self.volume)
        with self._aac_lock:
            hit = self._aac_cache.get(key)
        if hit is not None:
            return hit
        data = convert_file_to_aac_adts(
            src, sample_rate=self.sample_rate, volume=self.volume)
        with self._aac_lock:
            self._aac_cache[key] = data
            while len(self._aac_cache) > 32:
                self._aac_cache.pop(next(iter(self._aac_cache)))
        return data

    def _send_persistent(self, aac: bytes, channel: int) -> None:
        assert self._tunnel is not None
        try:
            host, port = self._tunnel.ensure()
            _visualtalk_once(
                aac, self.creds, host, port,
                channel=channel, timeout=self.timeout,
                sample_rate=self.sample_rate,
            )
            return
        except Exception:
            self._tunnel.invalidate()
            if not self.retry_persistent:
                raise
        # Thu lai 1 lan voi tunnel moi truoc khi bao loi.
        host, port = self._tunnel.ensure()
        try:
            _visualtalk_once(
                aac, self.creds, host, port,
                channel=channel, timeout=self.timeout,
                sample_rate=self.sample_rate,
            )
        except Exception:
            self._tunnel.invalidate()
            raise

    def __call__(self, audio_path: Path, channel: int | None = None) -> None:
        started = time.monotonic()
        use_channel = self.channel if channel is None else int(channel)
        try:
            if self._tunnel is not None:
                self._send_persistent(self._convert_cached(audio_path), use_channel)
            else:
                send_audio_file(
                    audio_path,
                    self.creds,
                    channel=use_channel,
                    timeout=self.timeout,
                    attempts=self.attempts,
                    retry_delay=self.retry_delay,
                    sample_rate=self.sample_rate,
                    volume=self.volume,
                )
            print(f"[Voice:P2P] Phat xong {Path(audio_path).name} "
                  f"({time.monotonic() - started:.1f}s).")
        except P2PTalkError as error:
            elapsed = time.monotonic() - started
            print(f"[Voice:P2P] Bo loi chao sau {elapsed:.1f}s: {error}")
            raise

    def close(self) -> None:
        if self._tunnel is not None:
            self._tunnel.close()


__all__ = [
    "ImouP2PCredentials",
    "ImouP2PTalkOutput",
    "P2PTalkError",
    "PersistentP2PTunnel",
    "convert_file_to_aac_adts",
    "find_ffmpeg",
    "send_audio_file",
]
