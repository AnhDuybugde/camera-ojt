"""Internal bearer authentication and short-lived browser stream URLs."""
import hashlib
import hmac
import os
import time
from urllib.parse import parse_qs, urlsplit


def signed_stream_url(base, path, ttl=120):
    key = os.getenv("CAMERA_INTERNAL_TOKEN", "")
    if not key:
        return base.rstrip("/") + path
    expires = int(time.time()) + ttl
    signature = hmac.new(key.encode(), f"{path}:{expires}".encode(), hashlib.sha256).hexdigest()
    return f"{base.rstrip('/')}{path}?expires={expires}&signature={signature}"


def authorized(target, authorization, key):
    if not key:
        return True  # Only permitted for loopback legacy processes.
    if hmac.compare_digest(authorization, "Bearer " + key):
        return True
    url = urlsplit(target)
    if url.path not in {"/cam_a.mjpg", "/cam_b.mjpg"}:
        return False
    query = parse_qs(url.query)
    try:
        expires = int(query["expires"][0])
        signature = query["signature"][0]
    except (KeyError, ValueError, IndexError):
        return False
    if not time.time() <= expires <= time.time() + 300:
        return False
    expected = hmac.new(key.encode(), f"{url.path}:{expires}".encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)
