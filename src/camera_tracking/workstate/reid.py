"""Re-ID appearance embedding (OSNet, GPU-first).

Production dùng OSNet body-ReID cho Global Identity continuity.
Không có fallback histogram: thà mất ID còn hơn nối nhầm người.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np


class EmbeddingExtractor(Protocol):
    def extract(self, crop_bgr: np.ndarray | None) -> np.ndarray | None: ...


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).ravel()
    right = np.asarray(right, dtype=np.float32).ravel()
    if left.shape != right.shape or left.size == 0:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


class OsnetEmbedding:
    """OSNet person-ReID adapter with lazy model loading (CUDA-first).

    Mặc định ``osnet_x0_25`` (lightweight, ~0.4M params) thay vì
    ``osnet_x1_0`` để chạy real-time 2 camera trên GPU tầm trung.
    Muốn chính xác hơn thì tune lên ``osnet_x0_5`` / ``osnet_x1_0``.
    """

    def __init__(self, model_name: str = "osnet_x0_25", device: str = "auto") -> None:
        self.model_name = model_name
        self.device_name = device
        self._model = None
        self._torch = None
        self._transform = None
        self._device = None
        self._use_half = False

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import torchreid
            from torchvision import transforms
        except ImportError as error:
            raise RuntimeError(
                "OSNet requires torch, torchvision and torchreid "
                "(pip install -e .[reid]). Abort instead of histogram fallback."
            ) from error
        # The default torch cache can be read-only in managed Windows
        # environments. Keep optional weights inside the project instead.
        cache_root = Path(os.environ.get("TORCH_HOME", "models/reid_cache"))
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(cache_root.resolve()))
        device = self.device_name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        model = torchreid.models.build_model(
            name=self.model_name, num_classes=1000, loss="softmax", pretrained=True
        )
        model.eval()
        # GPU: FP16 + cudnn benchmark giảm ~30-40% latency ReID.
        # CPU: giữ FP32.
        self._use_half = device == "cuda"
        if self._use_half:
            try:
                model = model.half()
            except (ValueError, RuntimeError):
                self._use_half = False
        model = model.to(device)
        if device == "cuda":
            try:
                torch.backends.cudnn.benchmark = True
            except (AttributeError, RuntimeError):
                pass
        self._torch = torch
        self._model = model
        self._device = torch.device(device)
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load(self) -> None:
        """Eagerly load weights so startup can report/fallback clearly."""
        self._load()

    def extract(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        self._load()
        import cv2

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        if self._use_half:
            tensor = tensor.half()
        with self._torch.inference_mode():
            embedding = self._model(tensor)
        vector = embedding.detach().float().cpu().numpy().ravel().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else None
