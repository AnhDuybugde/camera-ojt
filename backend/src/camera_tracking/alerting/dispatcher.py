"""Small, non-blocking notifier for confirmed attendance events."""
from __future__ import annotations

import json
import os
import queue
import threading
from collections.abc import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_EVENTS = frozenset({"CHECK_IN", "LEAVE_OFFICE"})


def _env_float(name: str, default: float, minimum: float) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


class AlertDispatcher:
    """Send queued events in a daemon thread; failed messages are not retried."""

    def __init__(self, webhook_url: str, token: str, *, timeout_s: float = 3.0) -> None:
        self.webhook_url = webhook_url
        self.token = token
        self.timeout_s = max(0.2, float(timeout_s))
        self.events: queue.Queue[dict | None] = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="n8n-alert-dispatcher", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.events.put_nowait(None)
        if self._thread is not None:
            self._thread.join(timeout=self.timeout_s + 1.0)

    def submit(self, payload: dict) -> bool:
        try:
            self.events.put_nowait(payload)
        except queue.Full:
            print("Alert dropped: in-memory queue is full.")
            return False
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            payload = self.events.get()
            if payload is None:
                return
            try:
                result = self._deliver(payload)
                if result.get("ignored"):
                    print(f"Alert ignored by n8n: {payload['event_type']}")
            except HTTPError as error:
                if error.code == 401:
                    print("Alert not sent: n8n rejected X-Camera-Token; check backend/.env and n8n IF node.")
                else:
                    print(f"Alert not sent: n8n returned HTTP {error.code}")
            except (URLError, OSError, TimeoutError, ValueError) as error:
                print(f"Alert not sent: {error}")

    def _deliver(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = Request(self.webhook_url, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "User-Agent": "camera-ojt-alerting/1",
            "X-Camera-Token": self.token,
        })
        with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 - configured operator endpoint
            if not 200 <= getattr(response, "status", 200) < 300:
                raise ValueError(f"n8n returned HTTP {response.status}")
            raw_response = response.read().decode("utf-8").strip()
        if not raw_response:
            return {"ok": True}
        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise ValueError("n8n returned invalid JSON") from error
        if result.get("ok") is not True:
            raise ValueError("n8n did not acknowledge the alert")
        return result


class AlertPublisher:
    def __init__(self, dispatcher: AlertDispatcher, enabled_events: Iterable[str]) -> None:
        self.dispatcher = dispatcher
        self.enabled_events = frozenset(str(item).strip().upper() for item in enabled_events if str(item).strip())

    @classmethod
    def from_env(cls, _output_dir: object = None) -> tuple["AlertPublisher | None", AlertDispatcher | None]:
        if os.getenv("ALERTING_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
            return None, None
        webhook_url = os.getenv("N8N_ALERT_WEBHOOK_URL", "").strip()
        token = os.getenv("N8N_ALERT_WEBHOOK_TOKEN", "").strip() or os.getenv("N8N_ALERT_WEBHOOK_SECRET", "").strip()
        if not webhook_url or not token:
            print("Alerting disabled: set N8N_ALERT_WEBHOOK_URL and N8N_ALERT_WEBHOOK_TOKEN.")
            return None, None
        raw_events = os.getenv("ALERTING_EVENTS", ",".join(sorted(DEFAULT_EVENTS)))
        events = [item for item in raw_events.split(",")]
        dispatcher = AlertDispatcher(
            webhook_url,
            token,
            timeout_s=_env_float("ALERTING_TIMEOUT_SECONDS", 3.0, 0.2),
        )
        return cls(dispatcher, events), dispatcher

    def publish(self, *, event_type: str, occurred_at: str, global_id: int, channel: str, person_id: str | None = None, person_name: str | None = None, confidence: float | None = None) -> bool:
        event_type = str(event_type).upper()
        if event_type not in self.enabled_events:
            return False
        return self.dispatcher.submit({
            "event_type": event_type,
            "person_name": str(person_name or person_id or "Người không xác định"),
            "occurred_at": str(occurred_at),
            "channel": str(channel),
        })
