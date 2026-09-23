"""Typed, dependency-free camera-ojt HTTP client and attendance bridge worker."""
from __future__ import annotations

import json
import os
import logging
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class CameraOjtClient:
    def __init__(self, base_url: str, supervisor_url: str = "", timeout: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.supervisor_url = supervisor_url.rstrip("/")
        self.timeout = timeout

    def _json(self, url: str, payload: dict | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url, data=data,
            headers={"Content-Type": "application/json", "User-Agent": "ai-mind-admin/1.0",
                     "Authorization": "Bearer " + os.getenv("CAMERA_INTERNAL_TOKEN", "")},
            method="POST" if data is not None else "GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            try:
                detail = json.load(error)
            except Exception:
                detail = {}
            raise RuntimeError(detail.get("message") or f"Backend HTTP {error.code}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError("Không kết nối được backend camera-ojt.") from error
        if not isinstance(result, dict):
            raise RuntimeError("Backend trả dữ liệu không hợp lệ.")
        return result

    def status(self) -> dict[str, Any]:
        return self._json(f"{self.base_url}/status.json")

    def supervisor_status(self) -> dict[str, Any]:
        if not self.supervisor_url:
            return {}
        return self._json(f"{self.supervisor_url}/api/status")

    def ask(self, question: str) -> dict[str, Any]:
        return self._json(f"{self.base_url}/assistant/ask", {"question": question})

    def stream_url(self, channel: str) -> str:
        safe = "b" if channel.upper() == "B" else "a"
        from camera_tracking.api.access import signed_stream_url
        return signed_stream_url(self.base_url, f"/cam_{safe}.mjpg")


class CameraOjtBridgeWorker:
    """Continuously moves camera attendance ticks into the dashboard SQLite DB."""

    def __init__(self, queue_path: Path, map_path: Path, service: Any, interval: float = 2.0) -> None:
        self.queue_path, self.map_path = Path(queue_path), Path(map_path)
        self.service, self.interval = service, max(0.5, interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_stats: dict[str, int] = {}
        self.error = ""

    def start(self) -> "CameraOjtBridgeWorker":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera-ojt-bridge")
        self._thread.start()
        return self

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:
        project_root = self.queue_path.parent.parent
        src = project_root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        try:
            from camera_tracking.store.aimind_bridge import drain_once, load_person_map
            from camera_tracking.store.queue import WriteQueue
            queue = WriteQueue(self.queue_path)
        except Exception as error:
            self.error = f"Không khởi tạo được bridge: {error}"
            logger.exception(self.error)
            return
        while not self._stop.is_set():
            try:
                self.last_stats = drain_once(queue, self.service, load_person_map(self.map_path))
                self.error = ""
            except Exception as error:
                self.error = str(error)
                logger.exception("camera-ojt bridge failed")
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 1)
