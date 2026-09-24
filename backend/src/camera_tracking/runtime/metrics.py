from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(slots=True)
class StageMetric:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    @property
    def mean_ms(self) -> float:
        return self.total_s * 1000.0 / self.count if self.count else 0.0


class StageMetrics:
    """Low-overhead stage timing for runtime diagnostics."""

    def __init__(self) -> None:
        self._stages: dict[str, StageMetric] = {}

    def observe(self, name: str, elapsed_s: float) -> None:
        """Record an already-measured duration without nesting a large block."""
        elapsed = max(0.0, float(elapsed_s))
        metric = self._stages.setdefault(name, StageMetric())
        metric.count += 1
        metric.total_s += elapsed
        metric.max_s = max(metric.max_s, elapsed)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - started)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "count": float(metric.count),
                "mean_ms": metric.mean_ms,
                "max_ms": metric.max_s * 1000.0,
            }
            for name, metric in self._stages.items()
        }


__all__ = ["StageMetric", "StageMetrics"]
