from __future__ import annotations

import io
import json
import pytest
import numpy as np

from camera_tracking.face.liveness import HttpLivenessGate, LivenessGate


def test_liveness_is_optional_in_monitor_mode(monkeypatch) -> None:
    monkeypatch.delenv("BIOMETRIC_LIVENESS_PROVIDER", raising=False)
    assert LivenessGate.from_env(required=False) is None


def test_liveness_fails_closed_when_required(monkeypatch) -> None:
    monkeypatch.delenv("BIOMETRIC_LIVENESS_PROVIDER", raising=False)
    with pytest.raises(RuntimeError):
        LivenessGate.from_env(required=True)


def test_http_liveness_requires_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("BIOMETRIC_LIVENESS_PROVIDER", "http")
    monkeypatch.delenv("BIOMETRIC_LIVENESS_URL", raising=False)
    with pytest.raises(RuntimeError):
        LivenessGate.from_env(required=True)


def test_remote_http_endpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        HttpLivenessGate(endpoint="http://example.com/verify")


def test_http_gate_applies_threshold(monkeypatch) -> None:
    class _Response(io.BytesIO):
        headers = {"Content-Length": "50"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    body = json.dumps({"live": True, "score": 0.91}).encode("utf-8")
    monkeypatch.setattr(
        "camera_tracking.face.liveness.urllib.request.urlopen",
        lambda request, timeout: _Response(body),
    )
    gate = HttpLivenessGate(
        endpoint="http://127.0.0.1:8890/verify", threshold=0.90
    )
    result = gate.verify(np.zeros((32, 32, 3), dtype=np.uint8))
    assert result.verified is True
    assert result.score == pytest.approx(0.91)
