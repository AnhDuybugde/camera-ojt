from __future__ import annotations

from typing import Protocol

import numpy as np

from camera_tracking.domain import Detection


class PersonDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...

