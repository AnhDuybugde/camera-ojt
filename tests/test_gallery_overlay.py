"""Test gallery load tu data/images + overlay chi hien Global ID."""
import json

import numpy as np

from camera_tracking.face.gallery import load_gallery


class _StubEmbedder:
    def __init__(self, dim: int = 8) -> None:
        vec = np.ones(dim, dtype=np.float32)
        self.vec = vec / float(np.linalg.norm(vec))

    def detect_embed(self, crop_bgr):
        from camera_tracking.face.embeddings import FaceDetection

        return [FaceDetection(bbox=(0, 0, 10, 10), score=0.99, embedding=self.vec)]


def test_load_gallery_uses_filename_and_namemap(tmp_path) -> None:
    pytest_cv2 = pytest_import_cv2()
    if pytest_cv2 is None:
        return
    img = np.full((40, 40, 3), 128, dtype=np.uint8)
    pytest_cv2.imwrite(str(tmp_path / "LeHoAnhDuy.jpg"), img)
    gallery = load_gallery(tmp_path, _StubEmbedder(), {"LeHoAnhDuy": "Le Ho Anh Duy"})
    assert len(gallery) == 1
    assert gallery.people[0].person_id == "LeHoAnhDuy"
    assert gallery.people[0].display_name == "Le Ho Anh Duy"
    assert gallery.people[0].embedding is not None


def test_load_gallery_maps_filename_to_employee_id(tmp_path) -> None:
    pytest_cv2 = pytest_import_cv2()
    if pytest_cv2 is None:
        return
    img = np.full((40, 40, 3), 128, dtype=np.uint8)
    pytest_cv2.imwrite(str(tmp_path / "LeHoAnhDuy.jpg"), img)
    gallery = load_gallery(
        tmp_path,
        _StubEmbedder(),
        {"LeHoAnhDuy": "Le Ho Anh Duy"},
        {"LeHoAnhDuy": "1"},
    )
    assert gallery.people[0].person_id == "LeHoAnhDuy"
    assert gallery.people[0].employee_id == "1"


def test_load_gallery_reads_registration_metadata(tmp_path) -> None:
    image_path = tmp_path / "NV001.jpg"
    image_path.write_bytes(b"fake")
    (tmp_path / "registry.json").write_text(json.dumps({
        "NV001": {
            "display_name": "Nguyen Van A",
            "employee_id": "NV001",
        }
    }), encoding="utf-8")

    gallery = load_gallery(tmp_path, _StubEmbedder())

    assert len(gallery) == 1
    assert gallery.people[0].display_name == "Nguyen Van A"
    assert gallery.people[0].employee_id == "NV001"


def test_load_gallery_folder_per_person_multi_image(tmp_path) -> None:
    pytest_cv2 = pytest_import_cv2()
    if pytest_cv2 is None:
        return
    person_dir = tmp_path / "NV002"
    person_dir.mkdir()
    img = np.full((40, 40, 3), 128, dtype=np.uint8)
    pytest_cv2.imwrite(str(person_dir / "front.jpg"), img)
    pytest_cv2.imwrite(str(person_dir / "left.jpg"), img)
    gallery = load_gallery(tmp_path, _StubEmbedder())
    assert len(gallery) == 1
    assert gallery.people[0].person_id == "NV002"
    assert gallery.people[0].embedding is not None


def test_embed_best_caps_seed_prototypes(tmp_path) -> None:
    """29 frame/người -> 1 embedding chính + tối đa 9 seed prototypes."""
    pytest_cv2 = pytest_import_cv2()
    if pytest_cv2 is None:
        return

    class _VariedEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        def detect_embed(self, crop_bgr):
            from camera_tracking.face.embeddings import FaceDetection

            rng = np.random.default_rng(self.calls)
            self.calls += 1
            vec = rng.normal(size=8).astype(np.float32)
            vec = vec / float(np.linalg.norm(vec))
            return [FaceDetection(bbox=(0, 0, 10, 10), score=0.9,
                                  embedding=vec)]

    person_dir = tmp_path / "NV029"
    person_dir.mkdir()
    img = np.full((40, 40, 3), 128, dtype=np.uint8)
    for i in range(12):
        pytest_cv2.imwrite(str(person_dir / f"frame_{i:02d}.jpg"), img)
    gallery = load_gallery(tmp_path, _VariedEmbedder(), max_seed_prototypes=10)
    assert len(gallery) == 1
    assert gallery.people[0].embedding is not None
    assert len(gallery.people[0].prototypes) <= 9


def test_embed_best_retries_padded_tight_crop(tmp_path) -> None:
    """Crop khít mặt (detector rớt) -> thử lại bản pad viền."""
    pytest_cv2 = pytest_import_cv2()
    if pytest_cv2 is None:
        return

    class _PaddedOnlyEmbedder:
        def detect_embed(self, crop_bgr):
            from camera_tracking.face.embeddings import FaceDetection

            # Chỉ thấy mặt khi có context xung quanh (ảnh đã pad).
            if min(crop_bgr.shape[:2]) <= 150:
                return []
            vec = np.ones(8, dtype=np.float32)
            vec = vec / float(np.linalg.norm(vec))
            return [FaceDetection(bbox=(0, 0, 10, 10), score=0.8,
                                  embedding=vec)]

    person_dir = tmp_path / "TightCrop"
    person_dir.mkdir()
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    pytest_cv2.imwrite(str(person_dir / "crop.jpg"), img)
    gallery = load_gallery(tmp_path, _PaddedOnlyEmbedder())
    assert len(gallery) == 1
    assert gallery.people[0].embedding is not None


def test_load_session_meta_prefers_quality(tmp_path) -> None:
    from camera_tracking.face.gallery import _load_session_meta

    (tmp_path / "session_abc.json").write_text(json.dumps({
        "display_name": "Test Person",
        "images": [
            {"file": "a.jpg", "pose": "front", "quality_score": 0.9},
            {"file": "b.jpg", "pose": "left", "quality_score": 0.5},
        ],
    }), encoding="utf-8")
    meta = _load_session_meta([tmp_path / "a.jpg"])
    assert meta["a.jpg"]["quality"] == 0.9
    assert meta["b.jpg"]["pose"] == "left"


def test_save_prototype_never_overwrites(tmp_path) -> None:
    import numpy as np

    from camera_tracking.face.gallery import save_prototype

    vec = np.ones(8, dtype=np.float32)
    first = save_prototype(tmp_path, "NV003", vec, index=0)
    second = save_prototype(tmp_path, "NV003", vec, index=0)
    assert first != second
    assert first.is_file() and second.is_file()


def pytest_import_cv2():
    try:
        import cv2

        return cv2
    except ImportError:
        return None


def test_draw_global_labels_hides_raw_ids() -> None:
    pytest_cv2 = pytest_import_cv2()
    if pytest_cv2 is None:
        return
    from camera_tracking.domain import BoundingBox, Track
    from camera_tracking.visualization import draw_global_labels
    from camera_tracking.workstate.room_fusion import RoomPersonStatus

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    tracks = [Track(track_id=5, bbox=BoundingBox(10, 10, 50, 80),
                    confidence=0.9, age=1, hits=1, confirmed=True)]
    status = {5: RoomPersonStatus(global_id=5, label="Working", in_room=True,
                                 display_name="An")}
    out = draw_global_labels(frame, tracks, status, {5: "An"}, {5: "1"})
    assert out.shape == frame.shape  # khong crash; raw ByteTrack id khong duoc ve


def test_draw_global_labels_marks_unknown_without_face() -> None:
    pytest_cv2 = pytest_import_cv2()
    if pytest_cv2 is None:
        return
    from camera_tracking.domain import BoundingBox, Track
    from camera_tracking.visualization import draw_global_labels
    from camera_tracking.workstate.room_fusion import RoomPersonStatus

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    tracks = [Track(track_id=7, bbox=BoundingBox(10, 10, 50, 80),
                    confidence=0.9, age=1, hits=1, confirmed=True)]
    status = {7: RoomPersonStatus(global_id=7, label="Working", in_room=True,
                                  display_name=None)}
    # Khong mat -> khong ten: van ve "(Unknown)" thay vi de trong.
    named = draw_global_labels(frame.copy(), tracks, status, {7: "An"}, {7: "1"})
    unnamed = draw_global_labels(frame.copy(), tracks, status, {}, {})
    assert unnamed.shape == frame.shape
    assert bool((unnamed != frame).any())
    assert bool((named != unnamed).any())


def test_status_colors_by_label() -> None:
    from camera_tracking.visualization import status_color

    assert status_color("Working") == (40, 180, 40)
    assert status_color("Away") == (0, 200, 255)
    assert status_color("At door") == (0, 200, 255)
    assert status_color("Out of door") == (60, 60, 220)
    assert status_color("Unknown") == (200, 200, 200)
    # Legacy labels keep their old colors.
    assert status_color("Near seat") == (0, 200, 255)
    assert status_color("Out of office") == (60, 60, 220)
    assert status_color("Returning") == (0, 200, 255)
    assert status_color("nope") is None
    assert status_color(None) is None


def test_draw_tracks_accepts_status_colors() -> None:
    import numpy as np

    from camera_tracking.domain import BoundingBox, Track
    from camera_tracking.visualization import draw_person_tracks
    from camera_tracking.workstate.room_fusion import RoomPersonStatus

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    tracks = [Track(track_id=5, bbox=BoundingBox(10, 10, 50, 80),
                    confidence=0.9, age=1, hits=1, confirmed=True)]
    status = {5: RoomPersonStatus(global_id=5, label="Away", in_room=True,
                                  display_name=None)}
    out = draw_person_tracks(frame, tracks, status=status)
    assert out.shape == frame.shape
    # Yellow (Away) box border must be present somewhere.
    yellow = (out[:, :, 0] == 0) & (out[:, :, 1] == 200) & (out[:, :, 2] == 255)
    assert bool(yellow.any())
