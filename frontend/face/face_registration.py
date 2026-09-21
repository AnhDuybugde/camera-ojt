"""Face registration validation and persistence."""
from __future__ import annotations

import time
from collections.abc import Callable

import cv2
import numpy as np

from config import settings
from auth.permissions import SYSTEM, require_owner
from database.db import Database
from face.detector import FaceDetector
from face.embedding import blur_score, embedding_to_blob, representative_embedding


class FaceRegistrationService:
    def __init__(self, db: Database, detector: FaceDetector, *, actor_role: str = SYSTEM, actor_employee_id: str | None = None) -> None:
        self.db = db
        self.detector = detector
        self.actor_role = actor_role
        self.actor_employee_id = actor_employee_id

    def _authorize(self, employee_id: str) -> None:
        require_owner(self.actor_role, employee_id, self.actor_employee_id)
        employee = self.db.get_employee(employee_id)
        if self.actor_role == "EMPLOYEE" and employee and employee["has_face"]:
            raise PermissionError("Đăng ký lại khuôn mặt cần quản lý thực hiện.")

    def validate_face(self, frame: np.ndarray, face: object) -> tuple[bool, str]:
        x1, y1, x2, y2 = map(int, face.bbox)
        if min(x2 - x1, y2 - y1) < settings.min_face_size:
            return False, "Move closer: the face is too small."
        h, w = frame.shape[:2]
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        sharpness = blur_score(crop)
        if sharpness < settings.blur_threshold:
            return False, (
                f"Face detected, but image is blurry "
                f"({sharpness:.1f}/{settings.blur_threshold:.1f}). Hold still and improve lighting."
            )
        if getattr(face, "embedding", None) is None:
            return False, "The model could not extract a face embedding."
        return True, "OK"

    def register_from_capture(
        self, employee_id: str, capture: object, samples: int | None = None,
        progress: Callable[[int, int, np.ndarray, str], None] | None = None,
        timeout_seconds: int = 60,
    ) -> np.ndarray:
        self._authorize(employee_id)
        employee = self.db.get_employee(employee_id)
        if not employee:
            raise ValueError("Employee does not exist.")
        if employee["has_face"]:
            raise ValueError("Employee already has a face embedding. Delete/replace it first.")
        target = samples or settings.registration_samples
        collected: list[np.ndarray] = []
        started, last_capture = time.monotonic(), 0.0
        while len(collected) < target and time.monotonic() - started < timeout_seconds:
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            faces = self.detector.detect(frame)
            message = "No face detected. Look toward the camera and move closer."
            if len(faces) > 1:
                message = f"Detected {len(faces)} faces. Only one person may be in frame."
            if len(faces) == 1:
                valid, message = self.validate_face(frame, faces[0])
                if valid and time.monotonic() - last_capture >= settings.registration_interval:
                    collected.append(np.asarray(faces[0].embedding, dtype=np.float32))
                    last_capture = time.monotonic()
                    message = "Captured - slowly change your head angle"
            if progress:
                progress(len(collected), target, frame, message)
        if len(collected) < target:
            raise TimeoutError(f"Captured only {len(collected)}/{target} valid samples before timeout.")
        embedding = representative_embedding(collected)
        blob, dimension = embedding_to_blob(embedding)
        self.db.save_embedding(employee_id, blob, dimension, actor_role=self.actor_role, actor_employee_id=self.actor_employee_id)
        return embedding

    def replace_from_embeddings(self, employee_id: str, embeddings: list[np.ndarray]) -> None:
        self._authorize(employee_id)
        if not self.db.get_employee(employee_id):
            raise ValueError("Employee does not exist.")
        blob, dimension = embedding_to_blob(representative_embedding(embeddings))
        self.db.save_embedding(employee_id, blob, dimension, actor_role=self.actor_role, actor_employee_id=self.actor_employee_id)

    def process_uploaded_image(
        self, image_bytes: bytes
    ) -> "tuple[bool, str, np.ndarray | None]":
        """Validate a single uploaded image and extract its embedding.

        Returns (ok, message_vi, embedding_or_None).
        Uses the same detection / validation pipeline as camera registration.
        """
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return False, "File ảnh không hợp lệ hoặc bị hỏng.", None

        faces = self.detector.detect(frame)
        if len(faces) == 0:
            return False, "Không phát hiện khuôn mặt trong ảnh.", None
        if len(faces) > 1:
            return False, (
                f"Phát hiện {len(faces)} khuôn mặt. "
                "Chỉ chấp nhận ảnh có đúng 1 người."
            ), None

        valid, msg = self.validate_face(frame, faces[0])
        if not valid:
            _vn: dict[str, str] = {
                "Move closer: the face is too small.":
                    "Khuôn mặt quá nhỏ trong ảnh.",
                "The model could not extract a face embedding.":
                    "Không thể trích xuất đặc trưng khuôn mặt.",
            }
            if msg.startswith("Face detected, but image is blurry"):
                msg = "Ảnh quá mờ. Hãy dùng ảnh rõ nét hơn."
            else:
                msg = _vn.get(msg, msg)
            return False, msg, None

        return True, "OK", np.asarray(faces[0].embedding, dtype=np.float32)

    def register_from_images(
        self,
        employee_id: str,
        image_bytes_list: list[bytes],
        overwrite: bool = False,
    ) -> dict:
        """Register (or replace) a face embedding from uploaded image bytes.

        Accepts 1 or more images. Rejects bad images, uses valid ones.
        Storage path is identical to camera registration (replace_from_embeddings).

        Returns dict: success, valid_count, rejected_count, rejected_reasons, message
        """
        self._authorize(employee_id)
        employee = self.db.get_employee(employee_id)
        if not employee:
            raise ValueError("Employee does not exist.")
        if employee["has_face"] and not overwrite:
            raise ValueError("ALREADY_HAS_FACE")

        valid_embeddings: list[np.ndarray] = []
        rejected_reasons: list[str] = []

        for raw in image_bytes_list:
            ok, msg, emb = self.process_uploaded_image(raw)
            if ok and emb is not None:
                valid_embeddings.append(emb)
            else:
                rejected_reasons.append(msg)

        if not valid_embeddings:
            return {
                "success": False,
                "valid_count": 0,
                "rejected_count": len(rejected_reasons),
                "rejected_reasons": rejected_reasons,
                "message": "Không có ảnh hợp lệ để đăng ký.",
            }

        self.replace_from_embeddings(employee_id, valid_embeddings)
        return {
            "success": True,
            "valid_count": len(valid_embeddings),
            "rejected_count": len(rejected_reasons),
            "rejected_reasons": rejected_reasons,
            "message": "Đăng ký khuôn mặt thành công.",
        }
