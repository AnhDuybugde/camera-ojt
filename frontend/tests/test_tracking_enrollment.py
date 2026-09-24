from __future__ import annotations

import json

from face import tracking_enrollment
from face.tracking_enrollment import TrackingEnrollmentClient


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return b'{"ok":true,"person_id":"NV009"}'


def test_tracking_enrollment_posts_employee_and_image(monkeypatch):
    captured = {}

    def fake_open(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(tracking_enrollment, "urlopen", fake_open)
    result = TrackingEnrollmentClient("http://camera:8765").register(
        "NV009", "Nhan vien moi", b"jpeg-data", overwrite=True
    )

    assert result["ok"] is True
    assert captured["url"] == "http://camera:8765/enrollment/register"
    assert captured["payload"]["employee_id"] == "NV009"
    assert captured["payload"]["display_name"] == "Nhan vien moi"
    assert captured["payload"]["consent"] is True
    assert captured["payload"]["overwrite"] is True
    assert captured["payload"]["image"] == "anBlZy1kYXRh"
