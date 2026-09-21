"""Local bridge from cached speech files to Imou WebSDK AudioTalk.

The Python process owns Open Platform credentials and the command queue. A
localhost-only browser page receives short-lived kit tokens and injects a Web
Audio MediaStream into ``imouPlayer.startTalk()``. Detection never talks to the
browser directly and therefore stays non-blocking.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import secrets
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import Request, urlopen


class AudioTalkError(RuntimeError):
    """Raised when the bridge cannot complete an AudioTalk command."""


class ImouApiError(AudioTalkError):
    """Raised when Imou Open Platform rejects a token request."""


@dataclass(frozen=True, slots=True)
class SpeechResult:
    command_id: str
    status: str
    detail: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"


@dataclass(slots=True)
class _PendingCommand:
    command_id: str
    audio_path: Path
    created_at: float
    expires_at: float
    delivered: bool = False
    event: threading.Event = field(default_factory=threading.Event)
    result: SpeechResult | None = None


#: Regional OpenAPI hosts, used by both Python token calls and the WebSDK
#: player (every player cloud call goes to `${domain}/openapi/...`, so an
#: empty domain makes live/talk fail silently with no playStart/talkStart).
OPENAPI_HOSTS = {
    "sg": "https://openapi-sg.easy4ip.com",
    "fk": "https://openapi-fk.easy4ip.com",
    "or": "https://openapi-or.easy4ip.com",
}


@dataclass(frozen=True, slots=True)
class ImouCredentials:
    app_id: str
    app_secret: str = field(repr=False)
    device_id: str
    device_code: str = field(repr=False)
    channel_id: str = "0"
    data_center: str = "sg"

    @property
    def openapi_host(self) -> str:
        return OPENAPI_HOSTS.get(
            (self.data_center or "sg").strip().lower(),
            "https://openapi-sg.easy4ip.com",
        )

    @classmethod
    def from_env(cls) -> ImouCredentials:
        names = ("IMOU_APP_ID", "IMOU_APP_SECRET", "IMOU_DEVICE_ID", "IMOU_DEVICE_CODE")
        values = {name: os.getenv(name, "").strip() for name in names}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise AudioTalkError(f"Missing Imou configuration: {', '.join(missing)}")
        return cls(
            app_id=values["IMOU_APP_ID"],
            app_secret=values["IMOU_APP_SECRET"],
            device_id=values["IMOU_DEVICE_ID"],
            device_code=values["IMOU_DEVICE_CODE"],
            channel_id=os.getenv("IMOU_CHANNEL_ID", "0").strip() or "0",
            data_center=os.getenv("IMOU_DATA_CENTER", "sg").strip() or "sg",
        )


class ImouTokenProvider:
    """Obtain and cache administrator and WebSDK kit tokens."""

    def __init__(self, credentials: ImouCredentials, timeout_s: float = 15.0) -> None:
        self.credentials = credentials
        self.timeout_s = timeout_s
        self._access_token = ""
        self._access_expires_at = 0.0
        self._kit_token = ""
        self._kit_expires_at = 0.0
        self._lock = threading.Lock()

    def kit_token(self, *, force_refresh: bool = False) -> str:
        with self._lock:
            now = time.monotonic()
            if not force_refresh and self._kit_token and now < self._kit_expires_at:
                return self._kit_token
            access_token = self._get_access_token(now)
            data = self._call(
                "getKitToken",
                {
                    "token": access_token,
                    "deviceId": self.credentials.device_id,
                    "channelId": self.credentials.channel_id,
                    "type": "0",
                },
            )
            token = str(data.get("kitToken", ""))
            if not token:
                raise ImouApiError("getKitToken returned no kitToken")
            lifetime = max(60, int(data.get("expireTime", 7200)))
            self._kit_token = token
            self._kit_expires_at = now + min(3600, max(60, lifetime - 300))
            return token

    def _get_access_token(self, now: float) -> str:
        if self._access_token and now < self._access_expires_at:
            return self._access_token
        data = self._call("accessToken", {})
        token = str(data.get("accessToken", ""))
        if not token:
            raise ImouApiError("accessToken returned no token")
        lifetime = max(60, int(data.get("expireTime", 259200)))
        self._access_token = token
        self._access_expires_at = now + max(60, lifetime - 600)
        return token

    def _call(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        stamp = int(time.time())
        nonce = str(uuid.uuid4())
        signature_text = (
            f"time:{stamp},nonce:{nonce},appSecret:{self.credentials.app_secret}"
        )
        payload = {
            "id": str(uuid.uuid4()),
            "system": {
                "ver": "1.0",
                "appId": self.credentials.app_id,
                "sign": hashlib.md5(signature_text.encode("utf-8")).hexdigest(),
                "time": stamp,
                "nonce": nonce,
            },
            "params": params,
        }
        host = self.credentials.openapi_host
        request = Request(
            f"{host}/openapi/{operation}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ImouApiError(f"Imou {operation} request failed: {error}") from error
        result = body.get("result", {})
        if str(result.get("code")) != "0":
            code = result.get("code", "unknown")
            message = result.get("msg", "request rejected")
            raise ImouApiError(f"Imou {operation} failed ({code}): {message}")
        return result.get("data", {}) or {}


class _BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], bridge: ImouAudioTalkBridge) -> None:
        self.bridge = bridge
        super().__init__(address, _BridgeHandler)


class _BridgeHandler(BaseHTTPRequestHandler):
    server: _BridgeServer

    def log_message(self, _format: str, *_args: object) -> None:
        return None

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        bridge = self.server.bridge
        split = urlsplit(self.path)
        if split.path == "/favicon.ico":
            # Browser tự xin icon; không cần auth, không phải lỗi.
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if split.path == "/":
            self._send_bytes(_BRIDGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if split.path == "/bridge.js":
            self._send_bytes(_BRIDGE_JS.encode("utf-8"), "text/javascript; charset=utf-8")
            return
        if split.path.startswith("/sdk/"):
            self._send_sdk_file(split.path.removeprefix("/sdk/"))
            return
        if split.path.startswith("/demo/"):
            # Demo chính chủ phục vụ chẩn đoán A/B trên cùng origin/port.
            self._send_sdk_file(split.path.removeprefix("/demo/"))
            return
        if not bridge._authorized(parse_qs(split.query)):
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if split.path == "/api/session":
            try:
                force_refresh = (parse_qs(split.query).get("refresh") or [""])[0] == "1"
                self._send_json(bridge._session_payload(force_refresh=force_refresh))
            except AudioTalkError as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_GATEWAY)
            return
        if split.path == "/api/debug":
            self._send_json(bridge._debug_payload())
            return
        if split.path == "/api/command":
            payload = bridge._next_command()
            if payload is None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            else:
                self._send_json(payload)
            return
        if split.path.startswith("/api/audio/"):
            command_id = unquote(split.path.removeprefix("/api/audio/"))
            path = bridge._audio_path(command_id)
            if path is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(path.read_bytes(), "audio/mpeg")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        bridge = self.server.bridge
        split = urlsplit(self.path)
        if split.path == "/api/debug" and bridge._authorized(parse_qs(split.query)):
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 2048)
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                bridge._note_debug(payload if isinstance(payload, dict) else {})
            except (ValueError, json.JSONDecodeError):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True})
            return
        if split.path != "/api/result" or not bridge._authorized(parse_qs(split.query)):
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 8192)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            bridge._complete(
                str(payload.get("commandId", "")),
                str(payload.get("status", "failed")),
                str(payload.get("detail", ""))[:500],
            )
        except (ValueError, json.JSONDecodeError):
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"ok": True})

    def _send_sdk_file(self, relative: str) -> None:
        root = self.server.bridge.sdk_dir
        try:
            path = (root / unquote(relative)).resolve()
            path.relative_to(root)
        except (OSError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send_bytes(path.read_bytes(), content_type)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _send_bytes(
        self,
        payload: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class ImouAudioTalkBridge:
    """Own the localhost server and serialize browser AudioTalk commands."""

    def __init__(
        self,
        *,
        credentials: ImouCredentials,
        sdk_dir: str | Path,
        host: str = "127.0.0.1",
        port: int = 8765,
        command_ttl_s: float = 5.0,
        talk_tail_s: float = 0.3,
        stream_id: int = 1,
        browser_executable: str | Path | None = None,
        launch_browser: bool = True,
        token_provider: ImouTokenProvider | None = None,
    ) -> None:
        self.credentials = credentials
        self.sdk_dir = Path(sdk_dir).expanduser().resolve()
        self.host = host
        self.port = port
        self.command_ttl_s = max(0.5, command_ttl_s)
        self.talk_tail_s = max(0.0, talk_tail_s)
        # 0 = main/HD, 1 = sub/SD (demo chính chủ mặc định 0; cam tắt
        # sub-stream thì streamId 1 live mãi không lên).
        self.stream_id = int(stream_id)
        self.browser_executable = Path(browser_executable) if browser_executable else None
        self.launch_browser = launch_browser
        self.token_provider = token_provider or ImouTokenProvider(credentials)
        self._secret = secrets.token_urlsafe(24)
        self._state_lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._debug_lock = threading.Lock()
        self._debug: dict[str, Any] = {}
        self._active: _PendingCommand | None = None
        self._server: _BridgeServer | None = None
        self._server_thread: threading.Thread | None = None
        self._browser: subprocess.Popen[bytes] | None = None

    @classmethod
    def from_env(
        cls,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        command_ttl_s: float = 5.0,
        talk_tail_s: float = 0.3,
        launch_browser: bool = True,
    ) -> ImouAudioTalkBridge:
        sdk_dir = os.getenv("IMOU_WEBSDK_DIR", "").strip()
        if not sdk_dir:
            bundled_local = Path(__file__).resolve().parents[3] / "output" / "imou_websdk"
            if bundled_local.is_dir():
                sdk_dir = str(bundled_local)
            else:
                raise AudioTalkError("Missing Imou configuration: IMOU_WEBSDK_DIR")
        browser = os.getenv("IMOU_BROWSER_EXECUTABLE", "").strip() or None
        try:
            stream_id = int(os.getenv("IMOU_STREAM_ID", "1") or 1)
        except ValueError:
            stream_id = 1
        return cls(
            credentials=ImouCredentials.from_env(),
            sdk_dir=sdk_dir,
            host=host,
            port=port,
            command_ttl_s=command_ttl_s,
            talk_tail_s=talk_tail_s,
            stream_id=stream_id,
            browser_executable=browser,
            launch_browser=launch_browser,
        )

    @property
    def url(self) -> str:
        if self._server is None:
            raise AudioTalkError("AudioTalk bridge has not started")
        return f"http://{self.host}:{self._server.server_port}/?key={self._secret}"

    def start(self) -> None:
        if self._server is not None:
            return
        required = (self.sdk_dir / "imou-player.js", self.sdk_dir / "WasmLib")
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise AudioTalkError(f"Incomplete IMOU_WEBSDK_DIR: {', '.join(missing)}")
        self._server = _BridgeServer((self.host, self.port), self)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="imou-audiotalk-http",
            daemon=True,
        )
        self._server_thread.start()
        if self.launch_browser:
            browser = self.browser_executable or _find_browser()
            self._browser = subprocess.Popen(
                [
                    str(browser),
                    f"--app={self.url}",
                    "--mute-audio",
                    "--autoplay-policy=no-user-gesture-required",
                    "--window-size=320,240",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._browser is not None and self._browser.poll() is None:
            self._browser.terminate()
        self._browser = None

    def play(self, audio_path: Path) -> SpeechResult:
        if self._server is None:
            raise AudioTalkError("AudioTalk bridge has not started")
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise AudioTalkError(f"Speech audio does not exist: {path}")
        with self._session_lock:
            now = time.monotonic()
            pending = _PendingCommand(
                command_id=uuid.uuid4().hex,
                audio_path=path,
                created_at=now,
                expires_at=now + self.command_ttl_s,
            )
            with self._state_lock:
                self._active = pending
            wait_s = self.command_ttl_s + 30.0
            pending.event.wait(wait_s)
            with self._state_lock:
                if self._active is pending:
                    self._active = None
            result = pending.result or SpeechResult(
                pending.command_id, "failed", "AudioTalk bridge response timeout"
            )
            if not result.succeeded:
                raise AudioTalkError(result.detail or result.status)
            return result

    def _authorized(self, query: dict[str, list[str]]) -> bool:
        supplied = (query.get("key") or [""])[0]
        return secrets.compare_digest(supplied, self._secret)
    def _session_payload(self, *, force_refresh: bool = False) -> dict[str, Any]:
        return {
            "deviceId": self.credentials.device_id,
            "deviceCode": self.credentials.device_code,
            "channelId": self.credentials.channel_id,
            "domain": self.credentials.openapi_host,
            "streamId": self.stream_id,
            "kitToken": self.token_provider.kit_token(force_refresh=force_refresh),
            "talkTailMs": round(self.talk_tail_s * 1000),
        }

    def _note_debug(self, payload: dict[str, Any]) -> None:
        with self._debug_lock:
            self._debug = {
                "liveReady": bool(payload.get("liveReady", False)),
                "lastPlayError": str(payload.get("lastPlayError", ""))[:300],
                "statusText": str(payload.get("statusText", ""))[:300],
                "reported_at": time.time(),
            }

    def _debug_payload(self) -> dict[str, Any]:
        with self._debug_lock:
            report = dict(self._debug)
        page_seen = bool(report)
        with self._state_lock:
            has_command = self._active is not None
        report["has_active_command"] = has_command
        report["page_seen"] = page_seen
        return report

    def debug_state(self) -> dict[str, Any]:
        """Fetch the latest page-reported state (for CLI diagnostics)."""
        if self._server is None:
            return {}
        url = (
            f"http://{self.host}:{self._server.server_port}"
            f"/api/debug?key={self._secret}"
        )
        try:
            with urlopen(url, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 - diagnostics must never break the CLI
            return {}

    def _next_command(self) -> dict[str, Any] | None:
        with self._state_lock:
            pending = self._active
            if pending is None or pending.delivered:
                return None
            if time.monotonic() > pending.expires_at:
                pending.result = SpeechResult(pending.command_id, "expired", "Greeting expired")
                pending.event.set()
                self._active = None
                return None
            pending.delivered = True
            return {
                "commandId": pending.command_id,
                "audioUrl": f"/api/audio/{pending.command_id}?key={self._secret}",
            }

    def _audio_path(self, command_id: str) -> Path | None:
        with self._state_lock:
            pending = self._active
            if pending is None or pending.command_id != command_id:
                return None
            return pending.audio_path

    def _complete(self, command_id: str, status: str, detail: str) -> None:
        with self._state_lock:
            pending = self._active
            if pending is None or pending.command_id != command_id:
                return
            pending.result = SpeechResult(command_id, status, detail)
            pending.event.set()


def _find_browser() -> Path:
    candidates = [
        Path(os.environ.get("PROGRAMFILES(X86)", ""))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", ""))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", ""))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Google/Chrome/Application/chrome.exe",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise AudioTalkError(
        "Edge/Chrome not found; set IMOU_BROWSER_EXECUTABLE to the browser executable"
    )


class ImouAudioTalkOutput:
    """Callable adapter used by ``VoiceGreeter``."""

    def __init__(self, bridge: ImouAudioTalkBridge) -> None:
        self.bridge = bridge

    def __call__(self, audio_path: Path, channel: int | None = None) -> None:
        # Bridge legacy chi co 1 loa: nhan kenh de dong protocol voi P2P.
        _ = channel
        self.bridge.play(audio_path)


_BRIDGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Imou AudioTalk Bridge</title>
<script type="module" src="/sdk/imou-player.js"></script>
<link rel="stylesheet" href="/sdk/imou-player.css">
<script defer src="/bridge.js"></script>
<style>#root{display:grid;width:1800px;height:900px;}#root-0{width:100%;height:100%;background:#000;}</style></head>
<body><div id="status">Starting Imou AudioTalk bridge...</div><div id="root"><div id="root-0"></div></div>
</body></html>
"""


_BRIDGE_JS = r"""
(() => {
  "use strict";
  window.__bridgeRuns = (window.__bridgeRuns || 0) + 1;
  console.log(`bridge script run #${window.__bridgeRuns}`);
  const secret = new URLSearchParams(location.search).get("key") || "";
  const api = (path) => `${path}${path.includes("?") ? "&" : "?"}key=${encodeURIComponent(secret)}`;
  const statusNode = document.getElementById("status");
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  let player = null;
  let session = null;
  let current = null;
  let liveReady = false;
  let lastPlayError = "";
  let talkStarted = false;
  let stopRequested = false;
  let talkTimer = null;
  let audioContext = null;
  let decodedAudio = null;
  let virtualDestinations = [];

  const nativeGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  const nativeEnumerateDevices = navigator.mediaDevices.enumerateDevices.bind(navigator.mediaDevices);

  navigator.mediaDevices.enumerateDevices = async () => {
    const devices = await nativeEnumerateDevices();
    if (current && !devices.some((item) => item.kind === "audioinput")) {
      return [...devices, {kind: "audioinput", deviceId: "imou-tts", label: "Imou TTS"}];
    }
    return devices;
  };

  navigator.mediaDevices.getUserMedia = async (constraints) => {
    if (!current || !constraints || !constraints.audio || constraints.video) {
      return nativeGetUserMedia(constraints);
    }
    audioContext ||= new AudioContext();
    await audioContext.resume();
    const destination = audioContext.createMediaStreamDestination();
    virtualDestinations.push(destination);
    return destination.stream;
  };

  async function postResult(status, detail = "") {
    if (!current) return;
    const commandId = current.commandId;
    await fetch(api("/api/result"), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({commandId, status, detail}),
    }).catch(() => {});
    cleanupCommand();
  }

  let lastReport = "";
  async function reportState(force = false) {
    const payload = JSON.stringify({
      liveReady,
      lastPlayError,
      statusText: statusNode.textContent || "",
    });
    if (!force && payload === lastReport) return;
    lastReport = payload;
    await fetch(api("/api/debug"), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: payload,
    }).catch(() => {});
  }

  function cleanupCommand() {
    clearTimeout(talkTimer);
    virtualDestinations.forEach((destination) => {
      destination.stream.getTracks().forEach((track) => track.stop());
    });
    virtualDestinations = [];
    decodedAudio = null;
    current = null;
    talkStarted = false;
    stopRequested = false;
  }

  function onPlayerEvent(event) {
    if (!event) return;
    if (event.type === "playStart") {
      liveReady = true;
      if (!current) statusNode.textContent = "Imou AudioTalk bridge ready (live playing)";
      reportState(true).catch(() => {});
      return;
    }
    if (!current) return;
    if (event.type === "talkStart") {
      talkStarted = true;
      clearTimeout(talkTimer);
      const destination = [...virtualDestinations].reverse().find(
        (item) => item.stream.getAudioTracks().some((track) => track.readyState === "live")
      );
      if (!destination || !decodedAudio) {
        player.stopTalk();
        postResult("failed", "No live virtual audio stream");
        return;
      }
      const source = audioContext.createBufferSource();
      source.buffer = decodedAudio;
      source.connect(destination);
      source.onended = () => {
        stopRequested = true;
        setTimeout(() => player.stopTalk(), session.talkTailMs || 300);
        talkTimer = setTimeout(
          () => postResult("failed", "talkEnd callback timeout"),
          (session.talkTailMs || 300) + 4000,
        );
      };
      source.start();
    } else if (event.type === "talkEnd" && stopRequested) {
      postResult("completed");
    }
  }

  async function onPlayerError(error) {
    const code = error && (error.errCode || error.errorCode || "unknown");
    const msg = error && (error.errMsg || error.message || "");
    if (!current) {
      // Live-play errors arrive with no active command: keep them for
      // /api/debug instead of swallowing, so CLI can show the real cause.
      lastPlayError = `live error ${code}${msg ? ": " + msg : ""}`;
      statusNode.textContent = `Bridge live loi: ${lastPlayError} (cho thu lai...)`;
      reportState(true).catch(() => {});
      return;
    }
    await postResult("failed", `Imou AudioTalk error ${code}`);
    if (["2001", "2006"].includes(String(code))) {
      await fetch(api("/api/session?refresh=1")).catch(() => {});
      location.reload();
    }
  }

  async function initialize() {
    // Script chạy dạng defer (sau parse, y demo indexEn.js): DOM + SDK sẵn
    // sàng, construct ngay sau khi có session, không sleep thêm.
    if (document.readyState !== "complete") {
      await new Promise((resolve) => {
        window.addEventListener("load", resolve, {once: true});
        setTimeout(resolve, 10000);
      });
    }
    const response = await fetch(api("/api/session"));
    if (!response.ok) throw new Error(`Session HTTP ${response.status}`);
    session = await response.json();
    // Thứ tự + kiểu giá trị mirror demo indexEn.js init() (verified demo OK
    // trên cùng server): chỉ khác id + thêm WasmLibPath/handleCallBack/
    // handleError phục vụ bridge. Autoplay mặc định, không play() thủ công.
    player = new imouPlayer({
      id: "root-0",
      domain: session.domain || "",
      deviceId: session.deviceId,
      token: session.kitToken,
      channelId: String(session.channelId ?? "0"),
      streamId: String(session.streamId ?? "1"),
      recordType: "cloud",
      handleStartTalk: () => {},
      name: 0,
      type: "1",
      code: session.deviceCode,
      kitToken: session.kitToken,
      WasmLibPath: "/sdk/",
      handleCallBack: onPlayerEvent,
      handleError: onPlayerError,
    });
    // Demo chính chủ KHÔNG gọi changeCode mà live vẫn lên (giải mã dùng
    // playToken/playTokenKey theo URL), nên không gọi để giữ init tối giản
    // hệt demo. Chỉ bật lại nếu cần cho camera mã hóa kiểu khác.
    // Phơi player cho chẩn đoán localhost (diag_voice.py đọc status trực
    // tiếp thay vì đoán mò qua sự kiện).
    window.__bridgePlayer = player;
    // Bẫy chẩn đoán: mặc định TẮT (nghi wrap manager.play/startPlay hoặc
    // getPlayUrl làm treo init WASM — treo câm, không lỗi). Chỉ bật khi mở
    // bridge với ?hook=1 để soi.
    const hooksOn = new URLSearchParams(location.search).get("hook") === "1";
    if (hooksOn) {
    try {
      const mgr0 = player.player;
      const hookMgr = (mgr) => {
        if (!mgr || mgr.__hooksDone) return;
        mgr.__hooksDone = true;
        if (typeof mgr.close === "function") {
          const origClose = mgr.close.bind(mgr);
          mgr.close = (...a) => {
            console.log(`manager.close called: ${new Error().stack || "no-stack"}`);
            return origClose(...a);
          };
        }
        if (typeof mgr.startPlay === "function") {
          const origSP = mgr.startPlay.bind(mgr);
          mgr.startPlay = (eid, t) => {
            console.log(`manager.startPlay called hasURL=${!!(t && t.streamURL)}`);
            return origSP(eid, t);
          };
        }
        if (typeof mgr.play === "function") {
          const origP = mgr.play.bind(mgr);
          mgr.play = (eid, t) => {
            console.log(`manager.play called hasURL=${!!(t && t.streamURL)}`);
            return origP(eid, t);
          };
        }
      };
      hookMgr(mgr0);
      // this.player có thể được gán lazy sau constructor: thử lại sau 1 tick.
      setTimeout(() => { try { hookMgr(player.player); } catch (_) {} }, 0);
    } catch (hookError) {
      console.log(`hook failed: ${hookError}`);
    }
    // Bẫy chẩn đoán: getPlayUrl treo hay trả rỗng đều hiện rõ trong console.
    try {
      const origGPU = player.getPlayUrl.bind(player);
      player.getPlayUrl = async () => {
        console.log("getPlayUrl: enter");
        try {
          const r = await origGPU();
          console.log(
            `getPlayUrl: exit url_len=${(r.playUrl || "").length} `
            + `streamType=${r.streamType} hasToken=${!!r.playToken}`);
          return r;
        } catch (gpuError) {
          console.log(`getPlayUrl threw: ${gpuError}`);
          throw gpuError;
        }
      };
    } catch (hookError) {
      console.log(`gpu hook failed: ${hookError}`);
    }
    } // end hooksOn
    // Không gọi player.play() thủ công: constructor đã autoplay.
    // Cold-start WASM (pthread + compile 5MB) từng mất ~40s mới Init Success:
    // chờ tới 90s, log mốc moduleKeys để soi timing.
    const t0 = Date.now();
    let loggedKeys = false;
    for (let waited = 0; waited < 90000; waited += 500) {
      if (liveReady) break;
      if (!loggedKeys && waited % 5000 < 500) {
        try {
          const keys = window.LCPlayModule
            ? Object.keys(window.LCPlayModule).length : -1;
          if (keys > 0) {
            loggedKeys = true;
            console.log(`module alive keys=${keys} at +${Date.now() - t0}ms`);
          }
        } catch (_) { /* chan doan: bo qua */ }
      }
      await sleep(500);
    }
    statusNode.textContent = liveReady
      ? "Imou AudioTalk bridge ready (live playing)"
      : "Imou AudioTalk bridge ready (dang cho live playStart...)";
    reportState(true).catch(() => {});
    if (!liveReady) {
      throw new Error(
        "Live khong playStart sau 90s: xem console live error"
      );
    }
  }

  async function acceptCommand(command) {
    current = command;
    if (!liveReady) {
      throw new Error(
        "Live view chua phat (thieu playStart): kiem tra kitToken, IMOU_CHANNEL_ID, camera online/WasmLib roi thu lai"
      );
    }
    audioContext ||= new AudioContext();
    await audioContext.resume();
    const response = await fetch(command.audioUrl);
    if (!response.ok) throw new Error(`Audio HTTP ${response.status}`);
    decodedAudio = await audioContext.decodeAudioData(await response.arrayBuffer());
    talkTimer = setTimeout(
      () => postResult(
        "failed",
        "talkStart callback timeout: startTalk bi SDK bo qua (live chua playing) hoac mic/socket P2P bi chan",
      ),
      20000,
    );
    player.startTalk();
  }

  async function poll() {
    let ticks = 0;
    for (;;) {
      try {
        if (!current) {
          const response = await fetch(api("/api/command"));
          if (response.status === 200) await acceptCommand(await response.json());
        }
        if (++ticks % 8 === 0) await reportState();
      } catch (error) {
        if (current) await postResult("failed", String(error));
        statusNode.textContent = `Bridge retrying: ${error}`;
      }
      await sleep(250);
    }
  }

  initialize().then(poll).catch(async (error) => {
    statusNode.textContent = `Bridge init failed: ${error}`;
    await sleep(3000);
    location.reload();
  });
})();
"""


__all__ = [
    "OPENAPI_HOSTS",
    "AudioTalkError",
    "ImouApiError",
    "ImouAudioTalkBridge",
    "ImouAudioTalkOutput",
    "ImouCredentials",
    "ImouTokenProvider",
    "SpeechResult",
]
