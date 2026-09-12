"""Face recognition cho diem danh channel B (cua ra vao).

Thiet ke:
- InsightFace `buffalo_s` lam backend mac dinh (detect + align + embed 512D).
- Mọi class nhan `embedder` dang inject duoc de test offline khong can model nang.
- Gallery nap tu `data/images/*.jpg|png`: stem file = person_id,
  ten hien thi qua `name_map` trong config neu co.
"""
from camera_tracking.face.attendance import AttendanceRecord, FaceAttendanceService
from camera_tracking.face.embeddings import (
    FaceDetection,
    FaceEmbedder,
    InsightFaceEmbedder,
    cosine_similarity,
    face_sharpness,
)
from camera_tracking.face.gallery import EnrolledPerson, FaceGallery, load_gallery
from camera_tracking.face.matcher import FaceMatcher, MatchResult

__all__ = [
    "AttendanceRecord",
    "EnrolledPerson",
    "FaceAttendanceService",
    "FaceDetection",
    "FaceEmbedder",
    "FaceGallery",
    "FaceMatcher",
    "InsightFaceEmbedder",
    "MatchResult",
    "cosine_similarity",
    "face_sharpness",
    "load_gallery",
]
