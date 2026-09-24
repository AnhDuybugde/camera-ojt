"""Client for registering web enrollment images with Camera OJT."""
from __future__ import annotations

import base64
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class TrackingEnrollmentError(RuntimeError):
    pass


class TrackingEnrollmentClient:
    def __init__(self, base_url: str, *, timeout: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def register(
        self,
        employee_id: str,
        display_name: str,
        image_bytes: bytes,
        *,
        overwrite: bool = False,
    ) -> dict:
        if not self.base_url:
            return {"ok": True, "skipped": True}
        payload = json.dumps({
            "employee_id": employee_id,
            "display_name": display_name,
            "image": base64.b64encode(image_bytes).decode("ascii"),
            "consent": True,
            "overwrite": bool(overwrite),
        }, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}/enrollment/register",
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "User-Agent": "ai-mind-enrollment/1",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                result = json.loads(response.read())
        except HTTPError as error:
            try:
                result = json.loads(error.read())
                message = result.get("message")
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                message = None
            raise TrackingEnrollmentError(
                str(message or f"Camera OJT từ chối đăng ký (HTTP {error.code}).")
            ) from error
        except (URLError, OSError) as error:
            raise TrackingEnrollmentError(
                "Không kết nối được Camera OJT. Hãy bật Camera AI rồi đăng ký lại."
            ) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise TrackingEnrollmentError(
                "Camera OJT trả về dữ liệu đăng ký không hợp lệ."
            ) from error
        if not isinstance(result, dict) or result.get("ok") is not True:
            message = result.get("message") if isinstance(result, dict) else None
            raise TrackingEnrollmentError(
                str(message or "Camera OJT không thể đăng ký khuôn mặt.")
            )
        return result
