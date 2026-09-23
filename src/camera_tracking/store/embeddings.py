"""Versioned enrollment samples; inference only reads the in-memory gallery.

The cache key includes image bytes and the complete embedding-space identifier.
Unversioned runtime prototypes are deliberately not imported here.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np


class EmbeddingStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS enrollment_samples (
                employee_id TEXT NOT NULL, source_hash TEXT NOT NULL,
                model_version TEXT NOT NULL, source_name TEXT NOT NULL,
                dimension INTEGER NOT NULL, vector BLOB NOT NULL,
                metadata TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(employee_id, source_hash, model_version))""")
            from camera_tracking.store.replication import ENROLLMENT_TABLES, initialize_replication
            initialize_replication(db, ENROLLMENT_TABLES)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    @staticmethod
    def checksum(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def get(self, employee_id: str, checksum: str, model_version: str):
        with self.connect() as db:
            row = db.execute(
                "SELECT vector, dimension FROM enrollment_samples "
                "WHERE employee_id=? AND source_hash=? AND model_version=?",
                (employee_id, checksum, model_version),
            ).fetchone()
        if row is None:
            return None
        vector = np.frombuffer(row[0], dtype="<f4").copy()
        return vector if vector.size == row[1] and np.isfinite(vector).all() else None

    def put(self, employee_id, checksum, model_version, source_name, vector, metadata, *, connection=None):
        vector = np.asarray(vector, dtype="<f4").ravel()
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm <= 1e-12:
            raise ValueError("Invalid enrollment embedding")
        vector = vector / norm
        def write(db):
            db.execute("""INSERT INTO enrollment_samples
                (employee_id, source_hash, model_version, source_name, dimension, vector, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(employee_id, source_hash, model_version) DO UPDATE SET
                source_name=excluded.source_name, metadata=excluded.metadata,
                vector=excluded.vector, dimension=excluded.dimension,
                updated_at=CURRENT_TIMESTAMP""",
                (employee_id, checksum, model_version, source_name, vector.size,
                 vector.tobytes(), json.dumps(metadata, ensure_ascii=False)))
        if connection is not None:
            write(connection)
        else:
            with self.connect() as db:
                write(db)

    def replace(self, employee_id, model_version, samples, *, connection=None):
        def write(db):
            db.execute("DELETE FROM enrollment_samples WHERE employee_id=? AND model_version=?",
                       (employee_id, model_version))
            for index, vector in enumerate(samples):
                self.put(employee_id, hashlib.sha256(vector.tobytes()).hexdigest(), model_version,
                         f"registration-{index}", vector, {"source": "registration"}, connection=db)
        if connection is not None:
            write(connection)
        else:
            with self.connect() as db:
                write(db)

    def samples(self, model_version):
        with self.connect() as db:
            rows = db.execute("SELECT employee_id, vector, metadata FROM enrollment_samples "
                              "WHERE model_version=? ORDER BY employee_id, source_hash", (model_version,)).fetchall()
        return [(employee, np.frombuffer(blob, dtype="<f4").copy(), json.loads(meta))
                for employee, blob, meta in rows]

    def summary(self):
        with self.connect() as db:
            return [dict(zip(("employee_id", "model_version", "samples"), row))
                    for row in db.execute("SELECT employee_id, model_version, COUNT(*) "
                                          "FROM enrollment_samples GROUP BY 1,2")]
