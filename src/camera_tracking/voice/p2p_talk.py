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


def convert_file_to_aac_adts(input_path: str | Path, *, sample_rate: int = 16000) -> bytes:
    """Convert TTS mp3 / WAV bat ky sang AAC-LC/ADTS mono 16 kHz (bytes)."""
    src = Path(input_path)
    if not src.is_file():
        raise P2PTalkError(f"Speech audio does not exist: {src}")
    ffmpeg = find_ffmpeg()
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
) -> None:
    """Convert + gui 1 file am thanh (mp3 TTS / wav) ra loa camera. Block."""
    creds = creds or ImouP2PCredentials.from_env()
    aac = convert_file_to_aac_adts(audio_path, sample_rate=sample_rate)
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
    ) -> None:
        self.creds = creds or ImouP2PCredentials.from_env()
        self.channel = int(channel)
        self.timeout = float(timeout)
        self.attempts = int(attempts)
        self.retry_delay = float(retry_delay)
        self.sample_rate = int(sample_rate)

    def __call__(self, audio_path: Path) -> None:
        started = time.monotonic()
        try:
            send_audio_file(
                audio_path,
                self.creds,
                channel=self.channel,
                timeout=self.timeout,
                attempts=self.attempts,
                retry_delay=self.retry_delay,
                sample_rate=self.sample_rate,
            )
        except P2PTalkError as error:
            elapsed = time.monotonic() - started
            print(f"[Voice:P2P] Bo loi chao sau {elapsed:.1f}s: {error}")
            raise


__all__ = [
    "ImouP2PCredentials",
    "ImouP2PTalkOutput",
    "P2PTalkError",
    "convert_file_to_aac_adts",
    "find_ffmpeg",
    "send_audio_file",
]
