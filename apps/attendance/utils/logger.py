"""Central logging setup."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config import settings


def setup_logging() -> None:
    settings.ensure_directories()
    root = logging.getLogger()
    if getattr(root, "_attendance_configured", False):
        return
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = RotatingFileHandler(
        settings.log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(console)
    root._attendance_configured = True  # type: ignore[attr-defined]
