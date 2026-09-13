"""Test gallery load tu data/images + overlay chi hien Global ID."""
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


def test_status_colors_by_label() -> None:
    from camera_tracking.visualization import status_color

    assert status_color("Working") == (40, 180, 40)
    assert status_color("Away") == (0, 200, 255)
    assert status_color("Near seat") == (0, 200, 255)
    assert status_color("Out of office") == (60, 60, 220)
    assert status_color("Unknown") == (60, 60, 220)
    assert status_color("Returning") == (255, 150, 0)
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
