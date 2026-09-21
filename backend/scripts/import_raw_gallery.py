"""Import enrollment multi-frame từ data/raw vào data/images gallery.

Mỗi folder data/raw/<Label>/ chứa 29 face-crop + session_*.json
(quality + pose từng frame). Script này:

- Map 4 người cũ về person_id hiện tại để giữ liên tục chấm công
  (Anh_Duy -> AnhDuy, Le_Van_Dai -> VanDai, Quoc_Ngoc -> QuocNgoc,
  Tran_Ba_Le_Hoang -> LeHoang). Người mới giữ nguyên tên folder.
- Copy toàn bộ frame vào data/images/<person_id>/ (giữ tên file gốc).
- Ghi/gộp data/images/registry.json {person_id: {display_name, employee_id}}
  (display_name từ session.json; employee của người cũ lấy theo
  face.employee_map trong config, người mới = person_id).
- Archive ảnh đơn legacy data/images/<person_id>.jp*g (trùng key) vào
  data/images/.legacy_backup/ để folder là nguồn duy nhất.

Chạy:
    python scripts/import_raw_gallery.py
    python scripts/import_raw_gallery.py --dry-run
    python scripts/import_raw_gallery.py --only Anh_Duy,Quoc_Ngoc --force
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from camera_tracking.config import load_config  # noqa: E402

# Giữ liên tục person_id/employee với gallery + attendance hiện tại.
KNOWN_PERSON_IDS = {
    "Anh_Duy": "AnhDuy",
    "Le_Van_Dai": "VanDai",
    "Quoc_Ngoc": "QuocNgoc",
    "Tran_Ba_Le_Hoang": "LeHoang",
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--gallery-dir", default=None,
                        help="Mặc định lấy face.gallery_dir trong config.")
    parser.add_argument("--only", default=None,
                        help="Chỉ import các folder này (cách nhau dấu phẩy).")
    parser.add_argument("--force", action="store_true",
                        help="Copy đè frame đã tồn tại.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ in kế hoạch, không ghi file.")
    return parser.parse_args()


def _read_session(folder: Path) -> tuple[str, dict[str, dict]]:
    """Trả (display_name, {filename: {quality, pose}})."""
    display = folder.name.replace("_", " ")
    meta: dict[str, dict] = {}
    sessions = sorted(folder.glob("session_*.json"))
    if not sessions:
        return display, meta
    try:
        payload = json.loads(sessions[0].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return display, meta
    if isinstance(payload.get("display_name"), str) and payload["display_name"].strip():
        display = " ".join(payload["display_name"].split())
    images = payload.get("images")
    if isinstance(images, list):
        for item in images:
            if isinstance(item, dict) and isinstance(item.get("file"), str):
                meta[item["file"]] = item
    return display, meta


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    gallery_dir = Path(args.gallery_dir or config.face.gallery_dir)
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        raise SystemExit(f"Không thấy thư mục raw: {raw_dir}")
    folders = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        folders = [p for p in folders if p.name in wanted]
        missing = wanted - {p.name for p in folders}
        if missing:
            raise SystemExit(f"Folder không tồn tại: {sorted(missing)}")
    if not folders:
        raise SystemExit("Không có folder nào để import.")

    registry_path = gallery_dir / "registry.json"
    registry: dict = {}
    if registry_path.is_file():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            registry = {}
    if not isinstance(registry, dict):
        registry = {}

    employee_map = dict(config.face.employee_map or {})
    backup_dir = gallery_dir / ".legacy_backup"

    for folder in folders:
        label = folder.name
        person_id = KNOWN_PERSON_IDS.get(label, label)
        display_name, _meta = _read_session(folder)
        frames = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        if not frames:
            print(f"Bỏ {label}: không có ảnh.")
            continue
        employee_id = employee_map.get(person_id, person_id)
        dest_dir = gallery_dir / person_id
        print(f"{label} -> {person_id} | {display_name} | "
              f"employee={employee_id} | {len(frames)} frames")
        if args.dry_run:
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src in frames:
            dest = dest_dir / src.name
            if dest.is_file() and not args.force:
                continue
            shutil.copy2(src, dest)
            copied += 1
        # Archive ảnh đơn legacy trùng key để folder là nguồn duy nhất.
        for ext in (".jpg", ".jpeg", ".png"):
            for legacy in (gallery_dir / f"{person_id}{ext}",
                           gallery_dir / f"{person_id}{ext.upper()}"):
                if legacy.is_file():
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    target = backup_dir / legacy.name
                    if target.exists():
                        target.unlink()
                    legacy.replace(target)
                    print(f"  archive legacy {legacy.name} -> .legacy_backup/")
        registry[person_id] = {
            "display_name": display_name,
            "employee_id": str(employee_id),
        }
        print(f"  copy {copied} frame mới vào {dest_dir.name}/")
    if args.dry_run:
        print("Dry-run: không ghi registry.")
        return
    gallery_dir.mkdir(parents=True, exist_ok=True)
    tmp = gallery_dir / ".registry.tmp.json"
    tmp.write_text(json.dumps(registry, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(registry_path)
    print(f"Registry: {registry_path} ({len(registry)} người)")


if __name__ == "__main__":
    main()
