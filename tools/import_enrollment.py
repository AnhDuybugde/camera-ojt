"""Import labeled enrollment folders; default is inventory only.

Run from the repository root with the project installed. No image is deleted.
Employee IDs come from the existing registry or exact existing display names;
otherwise a stable UUID-based code is created. Conflicting mappings stop import.
"""
import argparse
import json
from pathlib import Path
import sys
from uuid import uuid5, NAMESPACE_URL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    from camera_tracking.config import load_config
    from camera_tracking.face.gallery import load_registry, load_gallery
    cfg = load_config(ROOT / "config/default.yaml").face
    gallery_dir = ROOT / cfg.gallery_dir
    folders = sorted(p for p in gallery_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    counts = {p.name: sum(f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} for f in p.iterdir())
              for p in folders}
    print(json.dumps({"people": len(counts), "images": sum(counts.values()), "apply": args.apply}))
    if not args.apply:
        return 0
    from camera_tracking.application.backend import load_services
    from camera_tracking.face.embeddings import InsightFaceEmbedder
    from camera_tracking.store.embeddings import EmbeddingStore
    db, *_ = load_services()
    registry = load_registry(gallery_dir)
    employees = db.list_employees()
    for folder in folders:
        candidates = [e for e in employees if e["full_name"].strip().casefold() == folder.name.strip().casefold()]
        if len(candidates) > 1:
            raise ValueError("Ambiguous employee display name; fix registry mapping before import")
        registered = registry.get(folder.name, {}).get("employee_id", "").upper() or None
        employee_id = registered or (candidates[0]["employee_id"] if candidates else
                                    "NV-" + uuid5(NAMESPACE_URL, "camera-ojt:" + folder.name).hex[:12].upper())
        if registered and candidates and candidates[0]["employee_id"] != registered:
            raise ValueError("Registry conflicts with employee directory; reconcile before import")
        if db.get_employee(employee_id) is None:
            db.add_employee({"employee_id": employee_id, "full_name": folder.name})
        registry[folder.name] = {"employee_id": employee_id, "display_name": folder.name}
    registry_path = gallery_dir / "registry.json"
    if registry_path.exists():
        from datetime import datetime
        import shutil
        backup = ROOT / "var/backups" / ("registry-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(registry_path, backup)
    temporary = registry_path.with_suffix(".pending")
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(registry_path)
    embedder = InsightFaceEmbedder(cfg.model_pack, cfg.det_size, args.device)
    embedder._load()  # Missing dependencies/models must fail, never import an empty gallery.
    store = EmbeddingStore(db.path)
    with store.connect() as connection:
        connection.execute("UPDATE enrollment_samples SET employee_id=upper(employee_id) "
                           "WHERE employee_id LIKE 'NV-%' AND employee_id<>upper(employee_id)")
    version = f"{cfg.model_pack}:insightface-0.7:aligned-v1:det{cfg.det_size}"
    gallery = load_gallery(gallery_dir, embedder, sample_store=store, model_version=version,
                           refresh_samples=True)
    imported = 0
    for person in gallery.people:
        if person.embedding is not None and person.employee_id:
            vector = person.embedding.astype("float32")
            db.save_embedding(person.employee_id, vector.tobytes(), vector.size)
            imported += 1
    report = {"people_imported": imported, "people_expected": len(folders), "samples": store.summary()}
    report_path = ROOT / "var/enrollment-import.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"people_imported": imported, "report": str(report_path)}))
    return 0 if imported == len(folders) else 2


if __name__ == "__main__":
    raise SystemExit(main())
