from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

from camera_tracking.voice.imou_bridge import (
    ImouAudioTalkBridge,
    ImouCredentials,
    ImouTokenProvider,
)


class StubTokenProvider(ImouTokenProvider):
    def __init__(self, credentials: ImouCredentials) -> None:
        super().__init__(credentials)
        self.calls: list[str] = []

    def _call(self, operation, params):
        self.calls.append(operation)
        if operation == "accessToken":
            return {"accessToken": "access", "expireTime": 7200}
        return {"kitToken": f"kit-{len(self.calls)}", "expireTime": 7200}


def _credentials() -> ImouCredentials:
    return ImouCredentials("app", "secret", "device", "code")


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


def test_credentials_do_not_expose_secrets_in_repr() -> None:
    rendered = repr(_credentials())
    assert "secret" not in rendered
    assert "code" not in rendered


def test_token_provider_caches_and_force_refreshes() -> None:
    provider = StubTokenProvider(_credentials())
    assert provider.kit_token() == "kit-2"
    assert provider.kit_token() == "kit-2"
    assert provider.kit_token(force_refresh=True) == "kit-3"
    assert provider.calls == ["accessToken", "getKitToken", "getKitToken"]


def test_bridge_delivers_and_acknowledges_one_command(tmp_path: Path) -> None:
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "imou-player.js").write_text("", encoding="utf-8")
    (sdk / "WasmLib").mkdir()
    audio = tmp_path / "speech.mp3"
    audio.write_bytes(b"mp3")
    provider = StubTokenProvider(_credentials())
    bridge = ImouAudioTalkBridge(
        credentials=_credentials(), sdk_dir=sdk, port=0,
        launch_browser=False, token_provider=provider,
    )
    bridge.start()
    split = urlsplit(bridge.url)
    key = parse_qs(split.query)["key"][0]
    base = f"{split.scheme}://{split.netloc}"
    result_holder = []
    worker = threading.Thread(target=lambda: result_holder.append(bridge.play(audio)))
    try:
        worker.start()
        command = _get_json(f"{base}/api/command?{urlencode({'key': key})}")
        with urlopen(f"{base}{command['audioUrl']}", timeout=2) as response:
            assert response.read() == b"mp3"
        payload = json.dumps({
            "commandId": command["commandId"], "status": "completed", "detail": ""
        }).encode("utf-8")
        request = Request(
            f"{base}/api/result?{urlencode({'key': key})}", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urlopen(request, timeout=2) as response:
            assert response.status == 200
        worker.join(timeout=2)
        assert result_holder[0].succeeded
        session = _get_json(f"{base}/api/session?{urlencode({'key': key})}")
        assert session["deviceCode"] == "code"
        assert session["kitToken"] == "kit-2"
    finally:
        bridge.close()
