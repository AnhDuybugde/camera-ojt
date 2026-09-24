"""Face recognition cho diem danh channel B (cua ra vao).

Thiet ke:
- InsightFace `buffalo_s` lam backend mac dinh (detect + align + embed 512D).
- Mọi class nhan `embedder` dang inject duoc de test offline khong can model nang.
- Gallery nap tu `data/images/*.jpg|png`: stem file = person_id,
  ten hien thi qua `name_map` trong config neu co.
"""
from camera_tracking.face.attendance import AttendanceRecord, FaceAttendanceService
from camera_tracking.face.assignment import FaceCandidate, unique_face_assignments
from camera_tracking.face.consumer import FaceObservation, FaceTrackConsumer
from camera_tracking.face.embeddings import (
    FaceDetection,
    FaceEmbedder,
    InsightFaceEmbedder,
    cosine_similarity,
    face_sharpness,
    is_frontal_face,
)
from camera_tracking.face.gallery import EnrolledPerson, FaceGallery, load_gallery
from camera_tracking.face.liveness import HttpLivenessGate, LivenessGate, LivenessResult
from camera_tracking.face.matcher import FaceMatcher, MatchResult
from camera_tracking.face.reid import FaceReIDEmbedding
from camera_tracking.face.worker import FaceJob, FaceResult, FaceWorker

__all__ = [
    "AttendanceRecord",
    "EnrolledPerson",
    "FaceAttendanceService",
    "FaceCandidate",
    "FaceDetection",
    "FaceEmbedder",
    "FaceGallery",
    "FaceJob",
    "HttpLivenessGate",
    "FaceMatcher",
    "FaceObservation",
    "FaceReIDEmbedding",
    "FaceResult",
    "FaceTrackConsumer",
    "FaceWorker",
    "InsightFaceEmbedder",
    "LivenessGate",
    "LivenessResult",
    "MatchResult",
    "cosine_similarity",
    "face_sharpness",
    "is_frontal_face",
    "load_gallery",
    "unique_face_assignments",
]
