"""Co bao "loa dang phat" xuyen process (half-duplex mem).

Ben phat (greeter ffplay / Ha Linh paplay) goi mark_speaker_busy() khi
phat; ben nghe (capture_utterance) bo frame trong luc busy thay vi dem
hut wake / dot quota STT vo ich.

Dung timestamp-until (khong phai co boolean) de file cu do crash giua
chung tu het han, khong ket mic vinh vien.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

BUSY_MARGIN_S = 0.4
DEFAULT_BUSY_S = 6.0


def guard_file() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "output" / "qa_cache" / "speaker_busy.json"


def media_seconds(path: str | Path) -> float | None:
    """Do dai audio (giay); None khi khong do duoc."""
    try:
        import wave

        with wave.open(str(path), "rb") as wav:
            frames, rate = wav.getnframes(), wav.getframerate()
            if rate > 0 and frames > 0:
                return frames / rate
    except (OSError, EOFError, ValueError):
        pass
    try:
        import shutil
        import subprocess

        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            return None
        out = subprocess.run(
            [ffprobe, "-hide_banner", "-loglevel", "error",
             "-show_entries", "format=duration", "-of", "csv=p=0",
             str(path)],
            capture_output=True, text=True, timeout=5)
        value = float(out.stdout.strip())
        return value if value > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def mark_speaker_busy(path: str | Path | None = None,
                      extra_s: float = 0.8,
                      default_s: float = DEFAULT_BUSY_S,
                      hold_s: float | None = None) -> float:
    """Danh dau loa ban den timestamp; tra ve thoi diem het ban.

    hold_s: giu mic dung so giay nay (ghi de duration+extra). Dung khi
    biet chinh xac khoang can ne, VD talk() P2P: giu rong truoc phat
    (file + relay cham) roi that chat lai duoi vang sau khi phat xong.
    """
    if hold_s is not None:
        until = time.time() + max(0.0, float(hold_s))
    else:
        duration = media_seconds(path) if path else None
        until = time.time() + (duration or float(default_s)) + float(extra_s)
    try:
        target = guard_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f'{{"until": {until}}}', encoding="utf-8")
    except OSError:
        pass
    return until


def speaker_busy(now: float | None = None) -> bool:
    """True khi co process nao dang phat loa (trong han + margin)."""
    try:
        data = json.loads(guard_file().read_text(encoding="utf-8"))
        until = float(data.get("until", 0))
    except (OSError, ValueError, AttributeError):
        return False
    return (time.time() if now is None else now) < until + BUSY_MARGIN_S


__all__ = [
    "BUSY_MARGIN_S",
    "DEFAULT_BUSY_S",
    "guard_file",
    "mark_speaker_busy",
    "media_seconds",
    "speaker_busy",
]
