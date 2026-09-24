"""One-time Bé Xinh PCM cache builder.

Run this while run_workstate.py is stopped:
    python scripts/prewarm_hamy.py

It synthesizes the short phrases most likely to need instant playback. The
result is stored under output/hamy_audio_cache and reused by later runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from camera_tracking.audio import CameraCheckInAnnouncer
from camera_tracking.audio.announcer import PRIORITY_PREWARM
from camera_tracking.audio.companion import critical_prewarm_texts, prewarm_texts
from camera_tracking.audio.phrases import iter_prewarm_texts


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
    parser = argparse.ArgumentParser(description="Build Bé Xinh speech cache")
    parser.add_argument(
        "--critical-only",
        action="store_true",
        help="only build instant close-range and named wave responses",
    )
    args = parser.parse_args()

    speaker = CameraCheckInAnnouncer.from_env()
    if speaker is None:
        print("Không khởi tạo được Bé Xinh. Kiểm tra IMOU_* / IMOU_TALK_HELPER.")
        return 2

    people = _people()
    texts = (
        critical_prewarm_texts(people)
        if args.critical_only
        else prewarm_texts(people)
    )
    # Zone-event phrases share the same cache-first IMOU speaker. Include them
    # without removing any existing conversational/gesture lines.
    texts = list(dict.fromkeys([
        *texts,
        *iter_prewarm_texts(people, critical_only=args.critical_only),
    ]))

    mode = "critical" if args.critical_only else "full"
    print(
        f"[Bé Xinh/Prewarm] mode={mode} "
        f"people={len(people)} clips={len(texts)}"
    )
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
