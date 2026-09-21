"""One-time Bé Xinh PCM cache builder.

Run this while run_workstate.py is stopped:
    python scripts/prewarm_hamy.py

It synthesizes the short phrases most likely to need instant playback. The
result is stored under output/hamy_audio_cache and reused by later runs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from camera_tracking.audio import CameraCheckInAnnouncer
from camera_tracking.audio.announcer import PRIORITY_PREWARM
from camera_tracking.audio.companion import (
    GENERIC_ARRIVAL,
    GROUP_ARRIVAL,
    GENERIC_WAVE,
    GENERIC_APPROACH,
    GENERIC_STAND,
    GENERIC_WATER,
    GROUP_WATER,
    GENERIC_REST,
)


def _people() -> list[tuple[str, str]]:
    result: dict[str, str] = {}
    registry_path = PROJECT_ROOT / "data" / "images" / "registry.json"
    if registry_path.is_file():
        try:
            data = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(value, dict):
                    continue
                person_id = str(value.get("employee_id") or key).strip()
                display_name = " ".join(
                    str(value.get("display_name") or key).split()
                )
                if person_id and display_name:
                    result[person_id] = display_name

    # Fallback for older galleries that only use config face.name_map / employee_map.
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    if config_path.is_file():
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            config = {}
        face = config.get("face", {}) if isinstance(config, dict) else {}
        name_map = face.get("name_map", {}) if isinstance(face, dict) else {}
        employee_map = face.get("employee_map", {}) if isinstance(face, dict) else {}
        if isinstance(name_map, dict):
            for key, name in name_map.items():
                person_id = str(employee_map.get(key) or key).strip() if isinstance(employee_map, dict) else str(key)
                display_name = " ".join(str(name or key).split())
                if person_id and display_name:
                    result.setdefault(person_id, display_name)
    return sorted(result.items())


def main() -> int:
    speaker = CameraCheckInAnnouncer.from_env()
    if speaker is None:
        print("Không khởi tạo được Bé Xinh. Kiểm tra IMOU_* / IMOU_TALK_HELPER.")
        return 2

    people = _people()
    texts = [
        GENERIC_ARRIVAL,
        GROUP_ARRIVAL,
        GENERIC_WAVE,
        GENERIC_APPROACH,
        GENERIC_STAND,
        GENERIC_WATER,
        GROUP_WATER,
        GENERIC_REST,
    ]
    for _person_id, name in people:
        texts.extend(
            [
                f"{name} tới rồi nè! Bé Xinh chào nha.",
                f"{name} quay lại rồi nè! Bé Xinh chào nha.",
                f"Hihi, {name} chào Bé Xinh hả? Chào nha.",
                f"Ủa, {name} lại gần Bé Xinh hả?",
                f"Ơ, {name} đi đâu đó?",
                f"{name} ơi, uống miếng nước đi nha!",
                f"{name} ơi, ngồi lâu rồi. Duỗi người chút nha!",
            ]
        )

    # Preserve order while removing accidental duplicates.
    texts = list(dict.fromkeys(texts))

    print(f"[Bé Xinh/Prewarm] people={len(people)} clips={len(texts)}")
    added = speaker.prewarm(texts, priority=PRIORITY_PREWARM)
    print(f"[Bé Xinh/Prewarm] queued={added}. Đang tạo cache...")
    ok = speaker.wait_idle(timeout_s=1800.0)
    speaker.close()
    if not ok:
        print("[Bé Xinh/Prewarm] timeout; cache đã tạo được một phần.")
        return 1
    print("[Bé Xinh/Prewarm] xong. Lần chạy sau chào sẽ phát từ cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
