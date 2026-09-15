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


@dataclass(frozen=True, slots=True)
class ImouCredentials:
    app_id: str
    app_secret: str = field(repr=False)
    device_id: str
    device_code: str = field(repr=False)
    channel_id: str = "0"
    data_center: str = "sg"

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
        host = f"https://openapi-{self.credentials.data_center}.easy4ip.com"
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
        if split.path == "/":
            self._send_bytes(_BRIDGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if split.path == "/bridge.js":
            self._send_bytes(_BRIDGE_JS.encode("utf-8"), "text/javascript; charset=utf-8")
            return
        if split.path.startswith("/sdk/"):
            self._send_sdk_file(split.path.removeprefix("/sdk/"))
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
        self.browser_executable = Path(browser_executable) if browser_executable else None
        self.launch_browser = launch_browser
        self.token_provider = token_provider or ImouTokenProvider(credentials)
        self._secret = secrets.token_urlsafe(24)
        self._state_lock = threading.Lock()
        self._session_lock = threading.Lock()
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
        return cls(
            credentials=ImouCredentials.from_env(),
            sdk_dir=sdk_dir,
            host=host,
            port=port,
            command_ttl_s=command_ttl_s,
            talk_tail_s=talk_tail_s,
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
            "kitToken": self.token_provider.kit_token(force_refresh=force_refresh),
            "talkTailMs": round(self.talk_tail_s * 1000),
        }

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

    def __call__(self, audio_path: Path) -> None:
        self.bridge.play(audio_path)


_BRIDGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Imou AudioTalk Bridge</title>
<link rel="stylesheet" href="/sdk/imou-player.css"></head>
<body><div id="status">Starting Imou AudioTalk bridge...</div><div id="player"></div>
<script src="/sdk/imou-player.js"></script><script src="/bridge.js"></script></body></html>
"""


_BRIDGE_JS = r"""
(() => {
  "use strict";
  const secret = new URLSearchParams(location.search).get("key") || "";
  const api = (path) => `${path}${path.includes("?") ? "&" : "?"}key=${encodeURIComponent(secret)}`;
  const statusNode = document.getElementById("status");
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  let player = null;
  let session = null;
  let current = null;
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
    if (!event || !current) return;
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
    if (!current) return;
    const code = error && (error.errCode || error.errorCode || "unknown");
    await postResult("failed", `Imou AudioTalk error ${code}`);
    if (["2001", "2006"].includes(String(code))) {
      await fetch(api("/api/session?refresh=1")).catch(() => {});
      location.reload();
    }
  }

  async function initialize() {
    const response = await fetch(api("/api/session"));
    if (!response.ok) throw new Error(`Session HTTP ${response.status}`);
    session = await response.json();
    player = new imouPlayer({
      id: "player",
      width: 320,
      height: 180,
      deviceId: session.deviceId,
      channelId: Number(session.channelId || 0),
      token: session.kitToken,
      code: session.deviceCode,
      type: 1,
      streamId: 1,
      muted: true,
      controls: false,
      WasmLibPath: "/sdk/WasmLib/",
      handleCallBack: onPlayerEvent,
      handleError: onPlayerError,
    });
    await Promise.resolve(player.play());
    await sleep(1000);
    statusNode.textContent = "Imou AudioTalk bridge ready";
  }

  async function acceptCommand(command) {
    current = command;
    audioContext ||= new AudioContext();
    await audioContext.resume();
    const response = await fetch(command.audioUrl);
    if (!response.ok) throw new Error(`Audio HTTP ${response.status}`);
    decodedAudio = await audioContext.decodeAudioData(await response.arrayBuffer());
    talkTimer = setTimeout(
      () => postResult("failed", "talkStart callback timeout"),
      7000,
    );
    player.startTalk();
  }

  async function poll() {
    for (;;) {
      try {
        if (!current) {
          const response = await fetch(api("/api/command"));
          if (response.status === 200) await acceptCommand(await response.json());
        }
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
    "AudioTalkError",
    "ImouApiError",
    "ImouAudioTalkBridge",
    "ImouAudioTalkOutput",
    "ImouCredentials",
    "ImouTokenProvider",
    "SpeechResult",
]
