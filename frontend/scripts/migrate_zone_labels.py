"""One-time, idempotent migration for IMOU screenshot polygon coordinates."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any


def migrate_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    result = deepcopy(payload)
    changed = False
    cameras = result.get("cameras", {})
    if not isinstance(cameras, dict):
        raise ValueError("cameras must be an object")
    for camera in ("A", "B"):
        item = cameras.get(camera, {})
        size = item.get("image_size", [0, 0])
        width, height = int(size[0]), int(size[1])
        if width > 0 and height > 0:
            target_width = math.ceil(height * 16 / 9)
            padding = width - target_width
            if padding >= 40:
                left = padding // 2
                for zone in item.get("zones", []):
                    zone["points"] = [
                        [
                            round(max(0.0, min(1.0, (float(x) * width - left) / target_width)), 6),
                            round(float(y), 6),
                        ]
                        for x, y in zone.get("points", [])
                    ]
                item["image_size"] = [target_width, height]
                item["source_crop"] = {
                    "original_size": [width, height],
                    "left": left,
                    "top": 0,
                    "width": target_width,
                    "height": height,
                }
                changed = True
        for zone in item.get("zones", []):
            zone_id = str(zone.get("id", "")).strip().lower()
            label = str(zone.get("label", "")).strip().lower()
            if zone_id == "di_vao" or label == "đi vào":
                if zone.get("kind") != "door_inside":
                    zone["kind"] = "door_inside"
                    changed = True
            if zone.get("kind") in {"be_xinh", "water"}:
                if zone.get("privacy") != "standard":
                    zone["privacy"] = "standard"
                    changed = True
    return result, changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    path = args.path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    migrated, changed = migrate_payload(payload)
    if not changed:
        print("Zone labels already migrated.")
        return 0
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.stem}.pre-migration-{stamp}{path.suffix}")
    backup.write_bytes(path.read_bytes())
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(migrated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(f"Migrated: {path}")
    print(f"Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
