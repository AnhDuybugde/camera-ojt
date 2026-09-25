"""Enrollment inference runs on the backend, never in a browser session."""
import base64
import cv2
import numpy as np


class Enrollment:
    def __init__(self, detector):
        self.detector = detector

    def detect(self, jpeg):
        raw = base64.b64decode(jpeg, validate=True)
        if len(raw) > 5 * 1024 * 1024:
            raise ValueError("Image is too large")
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[0] * frame.shape[1] > 16_000_000:
            raise ValueError("Invalid image")
        return [{"bbox": face.bbox.tolist(), "embedding": face.embedding.tolist(),
                 "det_score": float(face.det_score)} for face in self.detector.detect(frame)]
