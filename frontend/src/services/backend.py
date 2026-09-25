"""Session-scoped API proxies. Streamlit never owns production DB workers."""
import json
import os
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

from backend.app.api.codec import decode, encode


class RemoteService:
    def __init__(self, service):
        self.service = service

    def __getattr__(self, method):
        if method.startswith("_"):
            raise AttributeError(method)

        def call(*args, **kwargs):
            return self.call(method, *args, **kwargs)
        return call

    def call(self, method, *args, **kwargs):
        url = os.getenv("CAMERA_API_URL", "http://127.0.0.1:8767").rstrip("/")
        request = Request(f"{url}/api/v1/{self.service}/{method}",
            data=json.dumps(encode({"args": args, "kwargs": kwargs})).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + st.session_state.get("api_token", "")})
        try:
            with urlopen(request, timeout=30) as response:
                result = decode(json.load(response)["result"])
        except HTTPError as error:
            payload = json.load(error)
            raise ValueError(payload.get("error", "Backend request failed")) from None
        except (URLError, TimeoutError):
            raise RuntimeError("Backend chưa chạy. Khởi động bằng camera-ojt run.") from None
        if self.service == "sync" and isinstance(result, dict):
            return SimpleNamespace(**result)
        return result


class RemoteRecognizer:
    def reload(self):
        # The backend refreshes the authoritative enrollment store.
        pass


class RemoteDetector:
    def detect(self, frame):
        import base64
        import cv2
        import numpy as np
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("Cannot encode enrollment image")
        faces = RemoteService("enrollment").detect(base64.b64encode(encoded).decode())
        return [SimpleNamespace(bbox=np.asarray(f["bbox"]),
                                embedding=np.asarray(f["embedding"], dtype=np.float32),
                                det_score=f["det_score"]) for f in faces]


class BackendWorker:
    """Compatibility notification target; worker lifecycle belongs to backend."""
    def request_sync(self):
        pass
