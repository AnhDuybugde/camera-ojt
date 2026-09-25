"""Compatibility shim — canonical home is backend.app.recognition.face_recognition."""
from backend.app.recognition.face_recognition import FaceRecognizer, Match

__all__ = ["FaceRecognizer", "Match"]
