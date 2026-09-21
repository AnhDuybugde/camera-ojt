from __future__ import annotations

import cv2
import numpy as np
import pytest

from database.db import Database
from face.face_registration import FaceRegistrationService


class DummyFace:
    def __init__(self, bbox=(10, 10, 100, 100), embedding=None):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.embedding = np.array(embedding if embedding is not None else [1.0, 0.0, 0.0], dtype=np.float32)


class MockDetector:
    def __init__(self, faces_to_return=None):
        self.faces_to_return = faces_to_return

    def detect(self, frame: np.ndarray):
        if self.faces_to_return is not None:
            return self.faces_to_return
        # Default: 1 valid face
        return [DummyFace()]


def _make_valid_image_bytes() -> bytes:
    img = np.zeros((150, 150, 3), dtype=np.uint8)
    # Draw some texture so blur_score passes
    cv2.putText(img, "TEST", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


def test_upload_single_valid_image(db):
    db.add_employee({"employee_id": "NV001", "full_name": "Nguyễn Văn A"})
    detector = MockDetector([DummyFace(bbox=(10, 10, 120, 120), embedding=[1.0, 0.0, 0.0])])
    service = FaceRegistrationService(db, detector)

    img_bytes = _make_valid_image_bytes()
    result = service.register_from_images("NV001", [img_bytes])

    assert result["success"] is True
    assert result["valid_count"] == 1
    assert result["rejected_count"] == 0
    emp = db.get_employee("NV001")
    assert emp["has_face"] == 1


def test_upload_multiple_images_with_averaging(db):
    db.add_employee({"employee_id": "NV002", "full_name": "Trần Thị B"})
    detector = MockDetector([DummyFace(bbox=(10, 10, 120, 120), embedding=[0.0, 1.0, 0.0])])
    service = FaceRegistrationService(db, detector)

    img1 = _make_valid_image_bytes()
    img2 = _make_valid_image_bytes()
    result = service.register_from_images("NV002", [img1, img2])

    assert result["success"] is True
    assert result["valid_count"] == 2
    assert result["rejected_count"] == 0
    assert db.get_employee("NV002")["has_face"] == 1


def test_upload_with_some_rejected_images(db):
    db.add_employee({"employee_id": "NV003", "full_name": "Lê C"})
    detector = MockDetector()
    service = FaceRegistrationService(db, detector)

    valid_img = _make_valid_image_bytes()
    corrupted_bytes = b"not an image file"

    result = service.register_from_images("NV003", [valid_img, corrupted_bytes])
    assert result["success"] is True
    assert result["valid_count"] == 1
    assert result["rejected_count"] == 1
    assert len(result["rejected_reasons"]) == 1


def test_upload_all_invalid_does_not_save(db):
    db.add_employee({"employee_id": "NV004", "full_name": "Phạm D"})
    # Detector returns 0 faces
    detector = MockDetector([])
    service = FaceRegistrationService(db, detector)

    img = _make_valid_image_bytes()
    result = service.register_from_images("NV004", [img])

    assert result["success"] is False
    assert result["valid_count"] == 0
    assert result["rejected_count"] == 1
    assert "Không có ảnh hợp lệ" in result["message"]
    assert db.get_employee("NV004")["has_face"] == 0


def test_upload_multiple_faces_rejected(db):
    detector = MockDetector([DummyFace(), DummyFace()])
    service = FaceRegistrationService(db, detector)

    img = _make_valid_image_bytes()
    ok, msg, emb = service.process_uploaded_image(img)
    assert ok is False
    assert "Chỉ chấp nhận ảnh có đúng 1 người" in msg
    assert emb is None


def test_upload_already_has_face_requires_overwrite(db):
    db.add_employee({"employee_id": "NV005", "full_name": "Hoàng E"})
    detector = MockDetector([DummyFace(bbox=(10, 10, 120, 120), embedding=[1.0, 0.0, 0.0])])
    service = FaceRegistrationService(db, detector)

    img = _make_valid_image_bytes()
    # First registration
    service.register_from_images("NV005", [img])

    # Second registration without overwrite flag
    with pytest.raises(ValueError) as excinfo:
        service.register_from_images("NV005", [img], overwrite=False)
    assert "ALREADY_HAS_FACE" in str(excinfo.value)

    # Second registration with overwrite flag
    res = service.register_from_images("NV005", [img], overwrite=True)
    assert res["success"] is True


def test_upload_nonexistent_employee(db):
    detector = MockDetector()
    service = FaceRegistrationService(db, detector)
    img = _make_valid_image_bytes()

    with pytest.raises(ValueError) as excinfo:
        service.register_from_images("NON_EXISTENT", [img])
    assert "Employee does not exist" in str(excinfo.value)

