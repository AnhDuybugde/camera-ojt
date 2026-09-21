"""Face device resolution: cuda fails fast without the GPU provider."""
from unittest.mock import patch

from camera_tracking.face.embeddings import InsightFaceEmbedder


def test_cuda_request_fails_fast_without_provider() -> None:
    embedder = InsightFaceEmbedder(device="cuda")
    with patch.object(
        InsightFaceEmbedder, "_cuda_provider_available", return_value=False
    ):
        try:
            embedder._load()
        except RuntimeError as error:
            assert "onnxruntime-gpu" in str(error)
        else:  # pragma: no cover - must not silently fall back
            raise AssertionError("expected RuntimeError for face_device=cuda")


def test_auto_prefers_cpu_without_provider() -> None:
    embedder = InsightFaceEmbedder(device="auto")
    assert embedder.device_name == "auto"
    assert embedder.resolved_device == "cpu"
