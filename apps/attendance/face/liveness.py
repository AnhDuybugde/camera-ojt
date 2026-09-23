"""Extensible liveness boundary; disabled by default."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import settings


@dataclass(slots=True)
class LivenessResult:
    is_live: bool
    score: float
    reason: str


class LivenessDetector:
    """Permissive adapter until a blink/head-motion/model strategy is configured."""

    def check(self, frame: np.ndarray, face: object) -> LivenessResult:
        if not settings.enable_liveness:
            return LivenessResult(True, 1.0, "Liveness check disabled")
        # Safe behavior: never claim liveness when the feature is enabled but no
        # vetted model has been integrated.
        return LivenessResult(False, 0.0, "No liveness strategy is configured")
