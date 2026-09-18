"""Khoa turn-taking "dang hoi-dap voi Ha Linh" xuyen process.

Ha Linh (subprocess) goi mark_turn() khi WAKE, clear_turn() khi ve cho.
Pipeline tracking (process chinh) goi turn_active() truoc khi xep loi
chao chu dong (vay tay) / tu dong (mat) de 2 loa khong noi chong nhau
giua cau hoi-dap.

Dung timestamp-until + TTL (khong phai boolean) de subprocess crash giua
turn thi khoa tu het han, chao khong bi cam vinh vien.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

DEFAULT_TURN_TTL_S = 90.0


def guard_file() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "output" / "qa_cache" / "voice_turn.json"


def mark_turn(ttl_s: float = DEFAULT_TURN_TTL_S) -> float:
    """Danh dau dang trong turn hoi-dap; tra timestamp het han."""
    until = time.time() + max(1.0, float(ttl_s))
    try:
        target = guard_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f'{{"until": {until}}}', encoding="utf-8")
    except OSError:
        pass
    return until


def clear_turn() -> None:
    """Ve cho: xoa khoa (chi xoa file minh tao)."""
    try:
        guard_file().unlink(missing_ok=True)
    except OSError:
        pass


def turn_active(now: float | None = None) -> bool:
    """True khi Ha Linh dang trong turn hoi-dap (khoa con han)."""
    try:
        data = json.loads(guard_file().read_text(encoding="utf-8"))
        until = float(data.get("until", 0))
    except (OSError, ValueError, AttributeError):
        return False
    return (time.time() if now is None else now) < until


__all__ = [
    "DEFAULT_TURN_TTL_S",
    "clear_turn",
    "guard_file",
    "mark_turn",
    "turn_active",
]
