"""Fail-closed liveness gate for production attendance.

A commercial deployment should connect an audited anti-spoof provider.  This
module intentionally does not pretend that a heuristic blink/head-turn check is
secure enough for payroll.  The built-in HTTP adapter accepts a face/person
crop, calls the configured provider, and normalizes its result.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
import urllib.request
from urllib.parse import urlparse

import cv2
import numpy as np


@dataclass(slots=True, frozen=True)
class LivenessResult:
    verified: bool
    score: float
    provider: str
    reason: str = ""


class LivenessGate:
    def verify(self, image_bgr: np.ndarray) -> LivenessResult:  # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def from_env(cls, *, required: bool = False) -> "LivenessGate | None":
        provider = os.getenv("BIOMETRIC_LIVENESS_PROVIDER", "").strip().lower()
        if not provider or provider in {"none", "disabled", "off"}:
            if required:
                raise RuntimeError("BIOMETRIC_LIVENESS_PROVIDER is required")
            return None
        if provider == "http":
            endpoint = os.getenv("BIOMETRIC_LIVENESS_URL", "").strip()
            if not endpoint:
                raise RuntimeError("BIOMETRIC_LIVENESS_URL is required for provider=http")
            return HttpLivenessGate(
                endpoint=endpoint,
                token=os.getenv("BIOMETRIC_LIVENESS_TOKEN", "").strip(),
                threshold=float(os.getenv("BIOMETRIC_LIVENESS_THRESHOLD", "0.80")),
                timeout_s=float(os.getenv("BIOMETRIC_LIVENESS_TIMEOUT_SECONDS", "1.5")),
            )
        raise RuntimeError(
            f"Unsupported BIOMETRIC_LIVENESS_PROVIDER={provider!r}; supported: http"
        )


class HttpLivenessGate(LivenessGate):
    """POST a JPEG to an audited liveness service.

    Expected response JSON (extra fields are ignored):
      {"live": true, "score": 0.97, "reason": "..."}
    """

    def __init__(
        self, *, endpoint: str, token: str = "", threshold: float = 0.80,
        timeout_s: float = 1.5,
    ) -> None:
        parsed = urlparse(endpoint)
        loopback_hosts = {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Liveness endpoint must be an absolute HTTP(S) URL")
        if parsed.scheme == "http" and parsed.hostname.lower() not in loopback_hosts:
            raise ValueError(
                "Remote liveness endpoint must use HTTPS; HTTP is allowed only on localhost"
            )
        self.endpoint = endpoint
        self.token = token
        self.threshold = min(1.0, max(0.0, float(threshold)))
        self.timeout_s = max(0.2, float(timeout_s))

    def verify(self, image_bgr: np.ndarray) -> LivenessResult:
        if image_bgr is None or not getattr(image_bgr, "size", 0):
            return LivenessResult(False, 0.0, "http", "empty image")
        ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return LivenessResult(False, 0.0, "http", "jpeg encode failed")
        payload = json.dumps({
            "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii")
        }).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            self.endpoint, data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                length = int(response.headers.get("Content-Length", "0") or 0)
                if length > 64 * 1024:
                    raise ValueError("provider response is too large")
                raw = response.read(64 * 1024 + 1)
                if len(raw) > 64 * 1024:
                    raise ValueError("provider response is too large")
                data = json.loads(raw.decode("utf-8"))
        except Exception as error:  # noqa: BLE001 - fail closed by design
            return LivenessResult(False, 0.0, "http", f"provider error: {error}")
        try:
            score = float(data.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        live_flag = bool(data.get("live", False))
        verified = live_flag and score >= self.threshold
        return LivenessResult(
            verified=verified,
            score=score,
            provider="http",
            reason=str(data.get("reason") or ("verified" if verified else "rejected")),
        )
