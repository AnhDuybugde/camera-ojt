"""Shared stdlib HTTP transport for backend domain routers."""
from __future__ import annotations

import json
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.app.api.codec import decode, encode


class ApiError(ValueError):
    """Backend answered 4xx/5xx (message already translated server-side)."""


class BackendOffline(RuntimeError):
    """Backend process is not reachable at CAMERA_API_URL."""


def connect(base_url: str, token: str = "", timeout: float = 30.0) -> Callable:
    """Return a ``call(service, method, *args, **kwargs)`` bound to a backend."""
    base = base_url.rstrip("/")

    def call(service: str, method: str, *args, **kwargs):
        request = Request(
            f"{base}/api/v1/{service}/{method}",
            data=json.dumps(encode({"args": args, "kwargs": kwargs})).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + token},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return decode(json.load(response)["result"])
        except HTTPError as error:
            try:
                payload = json.load(error)
            except Exception:
                payload = {}
            raise ApiError(payload.get("error") or f"Backend HTTP {error.code}") from None
        except (URLError, TimeoutError, OSError) as error:
            raise BackendOffline(
                "Backend chưa chạy. Khởi động bằng camera-ojt run.") from error

    return call
