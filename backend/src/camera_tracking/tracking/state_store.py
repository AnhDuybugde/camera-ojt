"""Small local store for day-scoped global identities."""
from __future__ import annotations

import io
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from camera_tracking.tracking.global_identity import GlobalIdentityManager

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_identities (
  day TEXT NOT NULL,
  global_id INTEGER NOT NULL,
  employee_id TEXT,
  gallery BLOB NOT NULL,
  model_key TEXT NOT NULL,
  PRIMARY KEY (day, global_id)
)
"""


@dataclass(slots=True)
class DailyIdentityStore:
    path: Path
    model_key: str

    def __init__(self, path: str | Path, model_key: str) -> None:
        self.path = Path(path)
        self.model_key = model_key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(_SCHEMA)

    def load(self, day: str, manager: GlobalIdentityManager) -> int:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT global_id, employee_id, gallery FROM daily_identities "
                "WHERE day = ? AND model_key = ? ORDER BY global_id",
                (day, self.model_key),
            ).fetchall()
        for global_id, employee_id, payload in rows:
            gallery = np.load(io.BytesIO(payload), allow_pickle=False)
            manager.restore_identity(
                int(global_id),
                employee_id=employee_id,
                gallery=[row for row in gallery],
            )
        return len(rows)

    def save(self, day: str, manager: GlobalIdentityManager) -> None:
        rows: list[tuple[str, int, str | None, bytes, str]] = []
        for global_id, identity in manager.identities.items():
            buffer = io.BytesIO()
            vectors = list(identity.gallery)
            width = vectors[0].size if vectors else 0
            array = np.stack(vectors) if vectors else np.empty((0, width), dtype=np.float32)
            np.save(buffer, array, allow_pickle=False)
            rows.append((day, global_id, identity.employee_id, buffer.getvalue(), self.model_key))
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM daily_identities WHERE day = ?", (day,))
            connection.executemany(
                "INSERT INTO daily_identities "
                "(day, global_id, employee_id, gallery, model_key) VALUES (?, ?, ?, ?, ?)",
                rows,
            )


__all__ = ["DailyIdentityStore"]
