"""Face embedder: wrap InsightFace, lazy-load, co fallback offline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class FaceDetection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 trong crop person
    score: float
    embedding: np.ndarray  # 512D L2-normalized (buffalo_s)


class FaceEmbedder(Protocol):
    def detect_embed(self, person_crop_bgr: np.ndarray | None) -> list[FaceDetection]: ...


def ensure_cuda_dlls() -> list[str]:
    """Dua site-packages/nvidia/*/bin vao DLL search path (Windows).

    May khong co CUDA Toolkit he thong van dung duoc CUDA/cuDNN tu pip
    wheels (nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, ...). Goi truoc
    khi tao InsightFace/onnxruntime session. Idempotent.
    """
    import os
    import sys

    added: list[str] = []
    roots: set[str] = set()
    try:
        import site as _site
        for path in (_site.getsitepackages() + [_site.getusersitepackages()]):
            if path:
                roots.add(path)
    except (AttributeError, ImportError, OSError):
        pass
    roots.update(sys.path)
    for root in sorted(roots):
        candidate = os.path.join(root, "nvidia")
        if not os.path.isdir(candidate):
            continue
        for pkg in sorted(os.listdir(candidate)):
            bin_dir = os.path.join(candidate, pkg, "bin")
            if not os.path.isdir(bin_dir) or bin_dir in added:
                continue
            try:
                os.add_dll_directory(bin_dir)
            except OSError:  # old runtime, PATH fallback below
                pass
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            added.append(bin_dir)
    return added


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).ravel()
    right = np.asarray(right, dtype=np.float32).ravel()
    if left.shape != right.shape or left.size == 0:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def face_sharpness(face_bgr: np.ndarray | None) -> float:
    """Do net bang variance of Laplacian. Tra 0.0 neu khong tinh duoc."""
    if face_bgr is None or face_bgr.size == 0:
        return 0.0
    try:
        import cv2
    except ImportError:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


class InsightFaceEmbedder:
    """Lazy-load InsightFace buffalo_s. Nhe voi caller khi chua can.

    Yeu cau: `pip install insightface onnxruntime`.
    Model se duoc tai lan dau (~300MB). Neu chua cai -> raise RuntimeError
    ro rang, caller co the bat va chay o che do khong-face.
    """

    def __init__(self, model_pack: str = "buffalo_s", det_size: int = 320,
                 device: str = "auto") -> None:
        self.model_pack = model_pack
        self.det_size = det_size
        self.device_name = (device or "auto").strip().lower()
        self.resolved_device = "cpu"
        self._app = None

    @staticmethod
    def _cuda_provider_available() -> bool:
        try:
            import onnxruntime as ort
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except ImportError:
            return False

    def _load(self):
        if self._app is not None:
            return self._app
        ensure_cuda_dlls()
        try:
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise RuntimeError(
                "insightface chua cai. Chay: pip install insightface onnxruntime "
                "(~300MB model tai lan dau)."
            ) from error
        want_cuda = self.device_name in ("auto", "cuda")
        use_cuda = want_cuda and self._cuda_provider_available()
        if self.device_name == "cuda" and not use_cuda:
            raise RuntimeError(
                "face_device=cuda nhung thieu CUDAExecutionProvider. "
                "Cai CUDA Toolkit 12.x + cuDNN roi: pip install onnxruntime-gpu. "
                "Tam dung face_device=cpu."
            )
        providers: list[str] | None = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if use_cuda else None
        app = FaceAnalysis(name=self.model_pack, providers=providers)
        # ctx_id=-1 => CPU; 0 => GPU dau tien (can onnxruntime-gpu).
        ctx_id = 0 if use_cuda else -1
        self.resolved_device = "cuda" if use_cuda else "cpu"
        try:
            app.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size))
        except TypeError:
            app.prepare(ctx_id=ctx_id)
        self._app = app
        return app

    @property
    def available(self) -> bool:
        try:
            self._load()
            return True
        except RuntimeError:
            return False

    def detect_embed(self, person_crop_bgr: np.ndarray | None) -> list[FaceDetection]:
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return []
        app = self._load()
        faces = app.get(person_crop_bgr) or []
        out: list[FaceDetection] = []
        for face in faces:
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                raw = getattr(face, "embedding", None)
                if raw is None:
                    continue
                emb = np.asarray(raw, dtype=np.float32)
                norm = float(np.linalg.norm(emb))
                if norm <= 1e-12:
                    continue
                emb = emb / norm
            bbox = tuple(float(v) for v in getattr(face, "bbox", [0, 0, 0, 0]))
            score = float(getattr(face, "det_score", 1.0))
            out.append(FaceDetection(bbox=bbox, score=score, embedding=np.asarray(emb)))
        # Uu tien mat ro nhat / diem cao nhat truoc.
        out.sort(key=lambda d: d.score, reverse=True)
        return out
