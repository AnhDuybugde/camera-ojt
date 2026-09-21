"""Enroll gallery tu thu muc `data/images/`."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from camera_tracking.face.embeddings import FaceEmbedder, cosine_similarity

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_REGISTRY_FILE = "registry.json"
_PROTOTYPES_DIR = ".prototypes"


@dataclass(slots=True)
class EnrolledPerson:
    person_id: str  # stem file, vi du "LeHoAnhDuy"
    display_name: str  # ten hien thi tren overlay/dashboard
    embedding: np.ndarray | None  # None neu chua tinh duoc (van hien ten khi fallback)
    source_path: str = ""
    employee_id: str | None = None
    prototypes: list[np.ndarray] = field(default_factory=list)


@dataclass(slots=True)
class FaceGallery:
    people: list[EnrolledPerson] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.people)

    def best_match(
        self, query: np.ndarray | None, threshold: float
    ) -> tuple[EnrolledPerson | None, float]:
        """Tra (person | None, score). Khong dat nguong -> None."""
        if query is None or not self.people:
            return None, 0.0
        ranked = self.ranked_matches(query)
        if not ranked:
            return None, 0.0
        best, best_score = ranked[0]
        if best is None or best_score < threshold:
            return None, max(best_score, 0.0)
        return best, float(best_score)

    def ranked_matches(
        self, query: np.ndarray | None
    ) -> list[tuple[EnrolledPerson, float]]:
        """Rank employees, using their best trusted prototype."""
        if query is None:
            return []
        ranked: list[tuple[EnrolledPerson, float]] = []
        for person in self.people:
            samples = ([person.embedding] if person.embedding is not None else [])
            samples.extend(person.prototypes)
            if samples:
                ranked.append(
                    (person, max(cosine_similarity(query, sample) for sample in samples))
                )
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def add_prototype(
        self,
        person_id: str,
        embedding: np.ndarray,
        *,
        seed_similarity: float,
        min_seed_similarity: float = 0.80,
        max_prototypes: int = 10,
        min_distance: float = 0.03,
    ) -> bool:
        """Grow a gallery only from a result already anchored to its seed."""
        if seed_similarity < min_seed_similarity:
            return False
        person = next((p for p in self.people if p.person_id == person_id), None)
        if person is None or person.embedding is None:
            return False
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return False
        vector = vector / norm
        samples = [person.embedding, *person.prototypes]
        if any(1.0 - cosine_similarity(vector, sample) < min_distance for sample in samples):
            return False
        person.prototypes.append(vector)
        if len(person.prototypes) > max(0, max_prototypes - 1):
            person.prototypes.pop(0)
        return True


def _read_image(path: Path) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def load_registry(gallery_dir: str | Path) -> dict[str, dict[str, str]]:
    """Load locally enrolled display metadata without making YAML mutable."""
    path = Path(gallery_dir) / _REGISTRY_FILE
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    registry: dict[str, dict[str, str]] = {}
    for person_id, value in payload.items():
        if not isinstance(person_id, str) or not isinstance(value, dict):
            continue
        display_name = value.get("display_name")
        employee_id = value.get("employee_id")
        if isinstance(display_name, str) and isinstance(employee_id, str):
            registry[person_id] = {
                "display_name": display_name,
                "employee_id": employee_id,
            }
    return registry


def load_gallery(
    gallery_dir: str | Path,
    embedder: FaceEmbedder | None,
    name_map: dict[str, str] | None = None,
    employee_map: dict[str, str] | None = None,
    max_seed_prototypes: int = 10,
) -> FaceGallery:
    """Load enrolled faces with multi-image support.

    Hỗ trợ 3 layout (tương thích ngược):
    - Legacy: ``data/images/<person_id>.jpg`` (1 ảnh/người).
    - Folder: ``data/images/<person_id>/*.jpg`` (3-5 góc/người).
    - Raw session: ``data/images/<person_id>/`` với 29 frame + file
      ``session_*.json`` (chất lượng + pose từng frame, xem
      ``scripts/import_raw_gallery.py``). Embedding chính = frame tốt nhất;
      các frame còn lại được chọn đa dạng (pose/góc khác nhau) thành seed
      prototypes để matching đa góc ngay từ đầu (không chờ live adapt).
    """
    root = Path(gallery_dir)
    gallery = FaceGallery()
    if not root.is_dir():
        return gallery
    registry = load_registry(root)
    registry_names = {
        person_id: value["display_name"] for person_id, value in registry.items()
    }
    registry_employees = {
        person_id: value["employee_id"] for person_id, value in registry.items()
    }
    name_map = {**registry_names, **(name_map or {})}
    employee_map = {**registry_employees, **(employee_map or {})}
    # Gom theo person_id từ cả 2 layout.
    buckets: dict[str, list[Path]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
            continue
        buckets.setdefault(path.stem, []).append(path)
    for path in sorted((root).glob("*")):
        if not path.is_dir() or path.name.startswith("."):
            continue
        images = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        if images:
            buckets.setdefault(path.name, []).extend(images)
    for person_id in sorted(buckets):
        paths = buckets[person_id]
        display_name = name_map.get(person_id, person_id)
        employee_id = employee_map.get(person_id)
        session_meta = _load_session_meta(paths)
        embedding, extra = _embed_best(
            paths, embedder,
            max_extra=max(0, max_seed_prototypes - 1),
            session_meta=session_meta,
        )
        prototypes = _load_prototypes(root, person_id)
        # Ảnh enroll phụ (ngoài ảnh tốt nhất) thành seed prototypes ngay.
        for vector in extra:
            _append_seed_prototype(
                prototypes, vector, max_prototypes=max_seed_prototypes)
        gallery.people.append(
            EnrolledPerson(
                person_id=person_id,
                display_name=display_name,
                embedding=embedding,
                source_path=str(paths[0]),
                employee_id=employee_id,
                prototypes=prototypes,
            )
        )
    return gallery


def _load_session_meta(paths: list[Path]) -> dict[str, dict]:
    """Đọc session_*.json cạnh các frame (nếu có) -> {filename: meta}.

    Mỗi meta có ``quality`` (0..1) và ``pose`` (front/left/right/up/down...).
    Không có file session -> {} (chấm điểm thuần bằng detector).
    """
    for path in paths:
        if path.name.startswith("session_") and path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return {}
            meta: dict[str, dict] = {}
            images = payload.get("images")
            if not isinstance(images, list):
                return {}
            for item in images:
                if not isinstance(item, dict):
                    continue
                name = item.get("file")
                if not isinstance(name, str):
                    continue
                try:
                    quality = float(item.get("quality_score", 0.0))
                except (TypeError, ValueError):
                    quality = 0.0
                pose = item.get("pose")
                meta[name] = {
                    "quality": max(0.0, min(1.0, quality)),
                    "pose": pose if isinstance(pose, str) else "",
                }
            return meta
    # Session nằm cùng folder nhưng không có trong bucket (đã lọc ảnh):
    # thử tìm cạnh frame đầu tiên.
    if paths:
        for candidate in sorted(paths[0].parent.glob("session_*.json")):
            return _load_session_meta([candidate, *paths])
    return {}


def _padded_copy(img: np.ndarray, ratio: float = 0.35) -> np.ndarray | None:
    """Pad viền replicate cho face-crop quá chặt (detector cần context).

    Crop 224x224 khít mặt thường rớt detection; pad 35% thì detect được
    (đã đo 8/8, score ~0.75). Trả None khi thiếu cv2.
    """
    try:
        import cv2
    except ImportError:
        return None
    height, width = img.shape[:2]
    pad = max(1, int(min(height, width) * ratio))
    return cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)


def _embed_best(
    paths: list[Path],
    embedder: FaceEmbedder | None,
    *,
    max_extra: int = 9,
    session_meta: dict[str, dict] | None = None,
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """Chọn embedding tốt nhất làm chính, các frame đa dạng làm seed.

    - Mỗi ảnh thử detect trực tiếp trước; rớt mới thử bản pad (giữ hành vi
      cũ cho ảnh full-scene, sửa crop chặt 224px).
    - Xếp hạng bằng detector score + kích thước mặt + quality session.
    - Extra được chọn tham lam theo pose luân phiên + khoảng cách cosine
      (>=0.03 so với vector đã chọn) để phủ nhiều góc, tối đa ``max_extra``.
    """
    if embedder is None:
        return None, []
    session_meta = session_meta or {}
    scored: list[tuple[float, str, np.ndarray]] = []
    for path in paths:
        if path.name.startswith("session_") and path.suffix == ".json":
            continue
        img = _read_image(path)
        if img is None:
            continue
        try:
            dets = embedder.detect_embed(img)
        except RuntimeError:
            dets = []  # insightface chua cai -> enroll chay che do ten-only
        if not dets:
            padded = _padded_copy(img)
            if padded is not None:
                try:
                    dets = embedder.detect_embed(padded)
                except RuntimeError:
                    dets = []
        for det in dets:
            width = det.bbox[2] - det.bbox[0]
            height = det.bbox[3] - det.bbox[1]
            size = min(width, height)
            if size <= 0:
                continue
            meta = session_meta.get(path.name, {})
            quality = float(meta.get("quality", 0.0))
            pose = str(meta.get("pose", ""))
            score = (
                float(det.score)
                + min(size, 200.0) / 1000.0
                + quality * 0.5
            )
            scored.append((
                score,
                pose,
                np.asarray(det.embedding, dtype=np.float32),
            ))
    if not scored:
        return None, []
    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0][2]
    # Round-robin theo pose để prototypes phủ nhiều góc nhìn.
    by_pose: dict[str, list[np.ndarray]] = {}
    for _, pose, vec in scored[1:]:
        by_pose.setdefault(pose or "", []).append(vec)
    ordered: list[np.ndarray] = []
    while any(by_pose.values()):
        for pose in sorted(by_pose):
            if by_pose[pose]:
                ordered.append(by_pose[pose].pop(0))
    extra: list[np.ndarray] = []
    chosen: list[np.ndarray] = [best]
    for vec in ordered:
        if len(extra) >= max(0, max_extra):
            break
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            continue
        unit = np.asarray(vec, dtype=np.float32).ravel() / norm
        if any(
            1.0 - cosine_similarity(unit, sample) < 0.03 for sample in chosen
        ):
            continue
        extra.append(unit)
        chosen.append(unit)
    return best, extra


def _append_seed_prototype(
    prototypes: list[np.ndarray], vector: np.ndarray, *, max_prototypes: int = 10
) -> None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return
    vector = np.asarray(vector, dtype=np.float32).ravel() / norm
    samples = list(prototypes)
    if any(1.0 - cosine_similarity(vector, s) < 0.03 for s in samples):
        return
    prototypes.append(vector)
    while len(prototypes) > max(0, max_prototypes - 1):
        prototypes.pop(0)


def _load_prototypes(root: Path, person_id: str) -> list[np.ndarray]:
    prototypes: list[np.ndarray] = []
    directory = root / _PROTOTYPES_DIR / person_id
    if not directory.is_dir():
        return prototypes
    for path in sorted(directory.glob("*.npy")):
        try:
            vector = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).ravel()
        except (OSError, ValueError):
            continue
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            prototypes.append(vector / norm)
    return prototypes


def save_prototype(
    gallery_dir: str | Path,
    person_id: str,
    embedding: np.ndarray,
    *,
    index: int,
) -> Path:
    """Persist one trusted embedding without storing another raw face image.

    Tên file tăng dần tới slot trống để không ghi đè prototype cũ khi
    gallery memory đã pop (circular buffer) nhưng file vẫn append-only.
    """
    directory = Path(gallery_dir) / _PROTOTYPES_DIR / person_id
    directory.mkdir(parents=True, exist_ok=True)
    slot = max(0, int(index))
    while True:
        path = directory / f"prototype_{slot:02d}.npy"
        if not path.exists():
            break
        slot += 1
        if slot > 9999:  #Practically unreachable; guard against infinite loop.
            break
    np.save(path, np.asarray(embedding, dtype=np.float32), allow_pickle=False)
    return path
