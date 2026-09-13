"""IMOU URL helpers without opening cameras at import time."""
from __future__ import annotations

import os
from urllib.parse import quote


def imou_url(channel: int, subtype: int = 1) -> str | None:
    """Build an IMOU RTSP URL from environment variables."""
    ip = os.getenv("IMOU_IP", "")
    user = os.getenv("IMOU_USER", "")
    password = os.getenv("IMOU_PASSWORD", "")
    if not ip or not user or not password:
        return None
    encoded_password = quote(password, safe="")
    return (
        f"rtsp://{user}:{encoded_password}@{ip}:554/"
        f"cam/realmonitor?channel={channel}&subtype={subtype}"
    )


def imou_urls(subtype: int = 1) -> tuple[str, str]:
    """Return both channel URLs or fail with an actionable message."""
    urls = [imou_url(channel, subtype) for channel in (1, 2)]
    if any(url is None for url in urls):
        raise RuntimeError("Thiếu IMOU_IP / IMOU_USER / IMOU_PASSWORD trong môi trường.")
    return urls[0], urls[1]  # type: ignore[return-value]


__all__ = ["imou_url", "imou_urls"]
