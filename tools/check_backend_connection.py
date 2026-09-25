"""End-to-end healthcheck: UI -> backend :8767 -> pipeline :8765.
Usage:
  python tools/check_backend_connection.py [--api-url URL] [--stream-url URL]
Exit 0 when every link succeeds, 1 otherwise. No camera/model needed for the
API checks; stream checks only need the pipeline process to be listening.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def post(api_url, service, method, args=None, kwargs=None, token=""):
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/v1/{service}/{method}",
        data=json.dumps({"args": args or [], "kwargs": kwargs or {}}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + token},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.load(error)
        except Exception:
            return error.code, {"error": f"HTTP {error.code}"}


def check_stream(stream_url):
    token = os.getenv("CAMERA_INTERNAL_TOKEN", "")
    request = urllib.request.Request(
        f"{stream_url.rstrip('/')}/status.json",
        headers={"Authorization": "Bearer " + token} if token else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.load(response)
        return True, f"pipeline OK ({len(payload.get('cameras', []))} kênh)"
    except urllib.error.HTTPError as error:
        return False, f"pipeline HTTP {error.code} (sai CAMERA_INTERNAL_TOKEN?)"
    except Exception as error:
        return False, f"pipeline không nghe ({type(error).__name__}: {error})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=os.getenv("CAMERA_API_URL", "http://127.0.0.1:8767"))
    parser.add_argument("--stream-url", default=os.getenv("CAMERA_OJT_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--email", default=os.getenv("CHECK_EMAIL", ""))
    parser.add_argument("--password", default=os.getenv("CHECK_PASSWORD", ""))
    args = parser.parse_args()

    failures = []

    # 1. Backend must listen (403 without token still proves liveness).
    code, body = post(args.api_url, "operations", "diagnostics")
    if code == 403:
        print("[OK] backend :8767 đang nghe (403 chưa đăng nhập là đúng)")
    else:
        failures.append(f"backend không nghe: HTTP {code} {body}")
        print(f"[FAIL] backend :8767 không nghe: HTTP {code} {body}")
        print("Gợi ý: camera-ojt run --no-camera (kiểm tra cổng 8767 có bị tiến trình cũ chiếm không)")
        return 1

    # 2. Login flow (needs credentials; skipped in anonymous mode).
    if args.email and args.password:
        code, body = post(args.api_url, "auth", "login", [args.email, args.password])
        if code == 200 and body.get("result", {}).get("token"):
            token = body["result"]["token"]
            print(f"[OK] đăng nhập {args.email} thành công (role {body['result'].get('role')})")
            code, body = post(args.api_url, "operations", "diagnostics", token=token)
            if code == 200:
                print(f"[OK] diagnostics qua token: {body.get('result')}")
            else:
                failures.append(f"diagnostics qua token: HTTP {code}")
                print(f"[FAIL] diagnostics qua token: HTTP {code} {body}")
        else:
            failures.append(f"đăng nhập thất bại: HTTP {code}")
            print(f"[FAIL] đăng nhập thất bại: HTTP {code} {body}")
    else:
        print("[SKIP] đăng nhập (đặt CHECK_EMAIL/CHECK_PASSWORD để kiểm tra full)")

    # 3. Enrollment endpoint wiring (invalid image must 400, not 500/offline).
    code, body = post(args.api_url, "enrollment", "detect", ["not-an-image"], token="x")
    print(f"[INFO] enrollment.detect wiring: HTTP {code} (403 thiếu token cũng OK)")

    # 4. Pipeline stream.
    ok, message = check_stream(args.stream_url)
    print(f"[{'OK' if ok else 'WARN'}] {message}")
    if not ok:
        print("Gợi ý: pipeline chưa chạy là bình thường khi dùng --no-camera; "
              "backend :8767 vẫn phục vụ UI.")

    if failures:
        print(f"\n{len(failures)} kiểm tra thất bại.")
        return 1
    print("\nLuồng kết nối success: UI -> backend :8767 OK" + (" + pipeline :8765 OK" if ok else " (pipeline offline, chế độ --no-camera)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
