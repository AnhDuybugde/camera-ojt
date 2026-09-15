"""Chẩn đoán voice bridge bằng Chromium thật (Playwright).

Mở trang bridge, thu console/pageerror, chờ live playStart rồi in toàn bộ
trạng thái. Dùng để tự查 tới khi thành công mà không cần nhìn màn hình.

Chạy:
    python scripts/diag_voice.py [--wait-s 30] [--talk]
    --talk: sau khi live ready, phát thử 1 câu TTS ra loa camera.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from camera_tracking.config import load_config
from camera_tracking.voice.greeter import VoiceGreeter
from camera_tracking.voice.imou_bridge import ImouAudioTalkBridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--wait-s", type=float, default=30.0)
    parser.add_argument("--talk", action="store_true",
                        help="Phát thử 1 câu sau khi live ready.")
    parser.add_argument("--headed", action="store_true",
                        help="Mở Chromium có giao diện (loại trừ lỗi headless).")
    parser.add_argument("--demo", action="store_true",
                        help="Chạy demo chính chủ indexEn.html thay vì trang bridge "
                             "(phân biệt lỗi code mình vs môi trường).")
    parser.add_argument("--demo-port", type=int, default=8768)
    parser.add_argument("--demo-autocreate", action="store_true",
                        help="Tạo player ngay lúc load, không click.")
    parser.add_argument("--demo-via-bridge", action="store_true",
                        help="Chạy demo chính chủ trên chính origin/port của "
                             "bridge (:8767/demo/) để isolate lỗi server.")
    parser.add_argument("--demo-bridge-args", action="store_true",
                        help="A/B quyết định: trong page demo đang chạy tốt, "
                             "construct thêm 1 player bằng args y hệt bridge "
                             "(container + params + WasmLibPath + callbacks) "
                             "xem có playStart không.")
    parser.add_argument("--demo-shell", default=None,
                        help="Bisect shell bridge trên page demo (cộng dần tới "
                             "khi gãy): trap | gum | preload | realgum. "
                             "VD --demo-shell=trap.")
    parser.add_argument("--channel", default=None,
                        help="Override IMOU_CHANNEL_ID cho lần chạy này.")
    parser.add_argument("--stream-id", default=None,
                        help="Override IMOU_STREAM_ID cho lần chạy này.")
    parser.add_argument("--text", default="Xin chào, đây là thử loa camera")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    load_dotenv(ROOT / ".env")
    if args.channel is not None:
        os.environ["IMOU_CHANNEL_ID"] = str(args.channel)
    if args.stream_id is not None:
        os.environ["IMOU_STREAM_ID"] = str(args.stream_id)
    config = load_config(args.config)
    voice = config.voice

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError("Chưa cài playwright: pip install playwright") from None

    if args.demo:
        _run_demo_mode(args)
        return

    bridge = ImouAudioTalkBridge.from_env(
        host=voice.bridge_host,
        port=voice.bridge_port,
        command_ttl_s=voice.command_ttl_s,
        talk_tail_s=voice.talk_tail_s,
        launch_browser=False,
    )
    logs: list[str] = []
    try:
        bridge.start()
        print(f"Bridge: {bridge.url.split('?')[0]}")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=not args.headed,
                args=[
                    "--mute-audio",
                    "--autoplay-policy=no-user-gesture-required",
                ])
            page = browser.new_page()
            page.on("console", lambda m: logs.append(
                f"[{m.type}] {m.text} ({m.location.get('url', '')})"
                if getattr(m, "location", None) else f"[{m.type}] {m.text}"))
            page.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
            def _on_response(r) -> None:
                try:
                    interesting = (
                        r.status >= 400 or "easy4ip" in r.url)
                    if not interesting:
                        return
                    logs.append(f"[http-{r.status}] {r.url}")
                except Exception as error:  # noqa: BLE001
                    logs.append(f"[resp-err] {error}")

            page.on("response", _on_response)
            page.on("requestfailed", lambda r: logs.append(
                f"[reqfail] {r.url} :: {r.failure}")
                if "ERR_ABORTED" not in str(r.failure) else None)

            def _on_request(req) -> None:
                try:
                    if "easy4ip" in req.url:
                        body = req.post_data or ""
                        # Che secret/token khi log.
                        body = re.sub(r'"kitToken"\s*:\s*"[^"]+"', '"kitToken":"<redacted>"', body)
                        body = body[:600]
                        logs.append(f"[req] {req.url.split('/')[-1]} :: {body}")
                    elif "/sdk/" in req.url or "WasmLib" in req.url or ".wasm" in req.url:
                        logs.append(f"[sdk-req] {req.url}")
                except Exception as error:  # noqa: BLE001
                    logs.append(f"[req-err] {error}")

            page.on("request", _on_request)
            page.goto(bridge.url)
            deadline = time.monotonic() + max(5.0, args.wait_s)
            ready = False
            while time.monotonic() < deadline:
                state = bridge.debug_state()
                if state.get("liveReady"):
                    ready = True
                    break
                if state.get("lastPlayError"):
                    break
                time.sleep(1.0)
            state = bridge.debug_state()
            print(f"debug_state: {state}")
            print(f"liveReady: {ready}")
            try:
                player_status = page.evaluate(
                    """() => {
                        const p = window.__bridgePlayer;
                        const glueScripts = Array.from(
                            document.querySelectorAll(
                                'script[src*="liblcplay"]')).map(
                                s => s.src.split('/').slice(-2).join('/'));
                        const wasmState = {
                            factory: typeof window.LCPlaySDK_Module,
                            moduleKeys: window.LCPlayModule
                                ? Object.keys(window.LCPlayModule).length : -1,
                            glueScripts,
                        };
                        if (!p) return {
                            exposed: false,
                            isolated: self.crossOriginIsolated,
                            cores: navigator.hardwareConcurrency,
                            ...wasmState,
                        };
                        const s = p.status || {};
                        const o = p.options || {};
                        const safe = (v) => typeof v === 'string'
                            ? (v.length > 12 ? `<str:${v.length}>` : v) : typeof v;
                        const host = document.getElementById('root-0');
                        return {
                            isolated: self.crossOriginIsolated,
                            cores: navigator.hardwareConcurrency,
                            factory: typeof window.LCPlaySDK_Module,
                            moduleKeys: window.LCPlayModule
                                ? Object.keys(window.LCPlayModule).length : -1,
                            glueScripts: Array.from(
                                document.querySelectorAll(
                                    'script[src*="liblcplay"]')).map(
                                    s => s.src.split('/').slice(-2).join('/')),
                            playerHTML: host ? host.innerHTML.length : -1,
                            playerKids: host ? host.querySelectorAll('*').length : -1,
                            canvasInPlayer: host ? host.querySelectorAll('canvas').length : -1,
                            videoInPlayer: host ? host.querySelectorAll('video').length : -1,
                            exposed: true,
                            playing: !!s.playing,
                            message: s.message || '',
                            codeSet: !!s.code,
                            streamId: s.streamId,
                            talk: !!s.talk,
                            talkProcessing: !!s.talkProcessing,
                            hasResolutions: !!(s.resolutions && s.resolutions.length),
                            videos: document.querySelectorAll('video').length,
                            canvas: document.querySelectorAll('canvas').length,
                            st_token: safe(s.token), st_device: safe(s.deviceId),
                            st_channel: safe(s.channelId), st_stream: safe(s.streamId),
                            st_domain: safe(s.domain),
                            op_token: safe(o.token), op_domain: safe(o.domain),
                            op_autoplay: o.autoplay,
                        };
                    }""")
                print(f"player: {player_status}")
            except Exception as error:  # noqa: BLE001
                print(f"player inspect lỗi: {error}")
            try:
                resources = page.evaluate(
                    """() => performance.getEntriesByType('resource')
                        .filter(e => /liblcplay|\.wasm|imou-player\.js/.test(e.name))
                        .map(e => ({
                            f: e.name.split('/').slice(-1)[0],
                            ts: Math.round(e.transferSize || 0),
                            body: Math.round(e.encodedBodySize || 0),
                            dur: Math.round(e.duration || 0),
                        }))""")
                print(f"sdk resources: {resources}")
            except Exception as error:  # noqa: BLE001
                print(f"resource inspect lỗi: {error}")
            try:
                probe = page.evaluate(
                    """() => new Promise((resolve) => {
                        const t = setTimeout(
                            () => resolve('TIMEOUT-15s'), 15000);
                        try {
                            window.LCPlaySDK_Module().then(
                                (m) => {
                                    clearTimeout(t);
                                    resolve('RESOLVED keys='
                                        + Object.keys(m || {}).length);
                                },
                                (e) => {
                                    clearTimeout(t);
                                    resolve('REJECTED ' + e);
                                });
                        } catch (e) {
                            clearTimeout(t);
                            resolve('THREW ' + e);
                        }
                    })""")
                print(f"module probe: {probe}")
            except Exception as error:  # noqa: BLE001
                print(f"module probe lỗi: {error}")
            if args.talk:
                if not ready:
                    print("Live chưa ready -> bỏ qua phát thử.")
                else:
                    greeter = VoiceGreeter(
                        cache_dir=voice.cache_dir, voice=voice.voice)
                    audio = greeter.synthesize(args.text)
                    if audio is None:
                        print("Không tạo được TTS.")
                    else:
                        try:
                            result = bridge.play(audio)
                            print(f"Camera playback: {result.status}")
                        except Exception as error:  # noqa: BLE001
                            print(f"Talk lỗi: {error}")
                            print(f"debug_state: {bridge.debug_state()}")
            browser.close()
    finally:
        bridge.close()
    print("---- console ----")
    if len(logs) > 60:
        for line in logs[:30]:
            print(line)
        print(f"... ({len(logs) - 60} dòng poll lặp lại đã ẩn) ...")
        for line in logs[-30:]:
            print(line)
    else:
        for line in logs:
            print(line)


class _CoopHandler(SimpleHTTPRequestHandler):
    """Static server cho SDK demo (cần COOP/COEP cho SharedArrayBuffer)."""

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, *_args: object) -> None:
        return None


def _run_demo_mode(args: argparse.Namespace) -> None:
    """Mở demo chính chủ indexEn.html với tham số thật, bấm Start, xem live."""
    from camera_tracking.voice.imou_bridge import (
        ImouCredentials,
        ImouTokenProvider,
    )

    credentials = ImouCredentials.from_env()
    token = ImouTokenProvider(credentials).kit_token()
    print(f"kitToken OK (len {len(token)}), channel={credentials.channel_id}")

    sdk_dir = ROOT / "output" / "imou_websdk"
    server = None
    if args.demo_via_bridge:
        from camera_tracking.voice.imou_bridge import ImouAudioTalkBridge

        config = load_config(args.config)
        voice = config.voice
        bridge = ImouAudioTalkBridge.from_env(
            host=voice.bridge_host,
            port=voice.bridge_port,
            command_ttl_s=voice.command_ttl_s,
            talk_tail_s=voice.talk_tail_s,
            launch_browser=False,
        )
        bridge.start()
        base = f"http://{voice.bridge_host}:{voice.bridge_port}/demo/indexEn.html"
    else:
        handler = partial(_CoopHandler, directory=str(sdk_dir))
        server = ThreadingHTTPServer(("127.0.0.1", args.demo_port), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{args.demo_port}/indexEn.html"
    logs: list[str] = []
    try:
        from playwright.sync_api import sync_playwright

        params = {
            "domain": credentials.openapi_host,
            "deviceId": credentials.device_id,
            "channelId": credentials.channel_id,
            "kitToken": token,
            "code": os.getenv("IMOU_DEVICE_CODE", ""),
            "streamId": os.getenv("IMOU_STREAM_ID", "1"),
            "type": "1",
        }
        url = f"{base}?{urlencode(params)}"
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=not args.headed,
                args=["--mute-audio",
                      "--autoplay-policy=no-user-gesture-required"])
            page = browser.new_page()
            page.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
            page.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))

            def _on_demo_request(req) -> None:
                try:
                    if "easy4ip" not in req.url:
                        return
                    body = req.post_data or ""
                    body = re.sub(r'"kitToken"\s*:\s*"[^"]+"', '"kitToken":"<redacted>"', body)
                    logs.append(f"[req] {req.url.split('/')[-1]} :: {body[:600]}")
                except Exception as error:  # noqa: BLE001
                    logs.append(f"[req-err] {error}")

            page.on("request", _on_demo_request)
            page.goto(url)
            if args.demo_shell:
                # Gài 1 mảnh shell bridge vào page demo TRƯỚC khi Start.
                piece = args.demo_shell
                if piece == "trap":
                    page.evaluate(
                        """() => {
                            let __f = undefined;
                            Object.defineProperty(
                                window, 'LCPlaySDK_Module', {
                                configurable: true,
                                get: () => __f,
                                set: (v) => {
                                    console.log('TRAP assigned');
                                    __f = (...a) => {
                                        console.log('TRAP called');
                                        const p = v(...a);
                                        Promise.resolve(p).then(
                                            (m) => console.log(
                                                'TRAP RESOLVED keys='
                                                + Object.keys(m || {}).length),
                                            (e) => console.log(
                                                'TRAP REJECTED ' + e));
                                        return p;
                                    };
                                },
                            });
                        }""")
                elif piece == "realgum":
                    # Exact override bản thật của bridge (current=null lúc
                    # init -> passthrough; chỉ active khi có lệnh talk).
                    page.evaluate(
                        """() => {
                            let current = null;
                            let audioContext = null;
                            let virtualDestinations = [];
                            const nativeGetUserMedia =
                                navigator.mediaDevices.getUserMedia.bind(
                                    navigator.mediaDevices);
                            const nativeEnumerateDevices =
                                navigator.mediaDevices.enumerateDevices.bind(
                                    navigator.mediaDevices);
                            navigator.mediaDevices.enumerateDevices =
                                async () => {
                                    const devices =
                                        await nativeEnumerateDevices();
                                    if (current && !devices.some(
                                        (item) =>
                                            item.kind === 'audioinput')) {
                                        return [...devices, {
                                            kind: 'audioinput',
                                            deviceId: 'imou-tts',
                                            label: 'Imou TTS'}];
                                    }
                                    return devices;
                                };
                            navigator.mediaDevices.getUserMedia =
                                async (constraints) => {
                                    if (!current || !constraints
                                        || !constraints.audio
                                        || constraints.video) {
                                        return nativeGetUserMedia(
                                            constraints);
                                    }
                                    audioContext ||= new AudioContext();
                                    await audioContext.resume();
                                    const destination = audioContext
                                        .createMediaStreamDestination();
                                    virtualDestinations.push(destination);
                                    return destination.stream;
                                };
                            console.log('SHELL realgum installed');
                        }""")
                elif piece == "preload":
                    page.evaluate(
                        """async () => {
                            await Promise.all([
                                fetch('/sdk/WasmLib/MultiThread/liblcplay.js',
                                    {cache: 'force-cache'}),
                                fetch('/sdk/WasmLib/MultiThread/liblcplay.worker.js',
                                    {cache: 'force-cache'}),
                                fetch('/sdk/WasmLib/MultiThread/liblcplay.wasm',
                                    {cache: 'force-cache'}),
                            ]);
                            console.log('SHELL preload done');
                        }""")
                    time.sleep(3.0)
                else:
                    raise RuntimeError(f"demo-shell lạ: {piece}")
                page.click("#start")
                deadline = time.monotonic() + max(5.0, args.wait_s)
                ok = False
                while time.monotonic() < deadline:
                    time.sleep(2.0)
                    try:
                        vs = page.evaluate(
                            """() => Array.from(
                                document.querySelectorAll('video')).map(
                                v => ({w: v.videoWidth, h: v.videoHeight}))""")
                    except Exception:  # noqa: BLE001
                        vs = []
                    ok = any((v.get("w", 0) or 0) > 0 for v in vs)
                    if ok:
                        break
                print(f"SHELL-{piece}: {'OK (live lên)' if ok else 'GÃY (treo)'}")
                print("---- console ----")
                for line in logs:
                    if ("TRAP" in line or "Init Success" in line
                            or "DecodeStart" in line or "SHELL" in line
                            or "Init Error" in line or "error" in line.lower()
                            or "NotAllowed" in line or "playStart" in line):
                        print(line)
                browser.close()
                return
                # Control: click Start (demo init, root-0). Test: player thứ 2
                # với args y hệt bridge trên div mới. Cùng document, cùng lúc.
                page.click("#start")
                time.sleep(1.0)
                page.evaluate(
                    """() => {
                        const q = new URLSearchParams(location.search);
                        window.__abEvents = [];
                        const host = document.createElement('div');
                        host.setAttribute('id', 'ab-player');
                        host.style.cssText =
                            'width:640px;height:480px;background:#000;';
                        document.body.appendChild(host);
                        window.__abPlayer = new imouPlayer({
                            id: 'ab-player',
                            domain: q.get('domain') || '',
                            deviceId: q.get('deviceId') || '',
                            token: q.get('kitToken') || '',
                            channelId: q.get('channelId') || '0',
                            streamId: q.get('streamId') || '1',
                            recordType: 'cloud',
                            handleStartTalk: () => {},
                            name: 0,
                            type: '1',
                            code: q.get('code') || '',
                            kitToken: q.get('kitToken') || '',
                            WasmLibPath: '/sdk/',
                            handleCallBack: (ev) => {
                                window.__abEvents.push(
                                    ev && ev.type ? ev.type : 'unknown');
                                console.log(`AB-EVENT ${(ev && ev.type) || '?'}`);
                            },
                            handleError: (err) => console.log(
                                `AB-ERROR ${JSON.stringify(err || {})}`),
                        });
                    }""")
                deadline = time.monotonic() + max(5.0, args.wait_s)
                while time.monotonic() < deadline:
                    time.sleep(2.0)
                    try:
                        ab = page.evaluate(
                            """() => ({
                                ctlVideos: Array.from(
                                    document.querySelectorAll(
                                        '#root-0 video')).map(
                                        v => ({w: v.videoWidth,
                                              h: v.videoHeight})),
                                abEvents: window.__abEvents || [],
                                abVideos: Array.from(
                                    document.querySelectorAll(
                                        '#ab-player video')).map(
                                        v => ({w: v.videoWidth,
                                              h: v.videoHeight})),
                            })""")
                    except Exception:  # noqa: BLE001
                        ab = {}
                    print(f"A/B: {ab}")
                    ctl_ok = any(
                        (v.get("w", 0) or 0) > 0
                        for v in ab.get("ctlVideos", []))
                    ab_ok = any(
                        (v.get("w", 0) or 0) > 0
                        for v in ab.get("abVideos", []))
                    if ctl_ok and (ab_ok or "playStart" in ab.get("abEvents", [])):
                        break
                print(f"A/B RESULT control_ok={ctl_ok} ab_ok={ab_ok} "
                      f"events={ab.get('abEvents', [])}")
                print("---- console ----")
                for line in logs:
                    if ("AB-" in line or "Init Success" in line
                            or "DecodeStart" in line or "Init Error" in line
                            or "ERROR" in line or "error" in line.lower()):
                        print(line)
                browser.close()
                return
            if args.demo_autocreate:
                # Tạo player ngay lúc load, không click (mô phỏng bridge).
                page.evaluate(
                    """() => {
                        const q = new URLSearchParams(location.search);
                        init('root-0', {
                            name: 0,
                            domain: q.get('domain') || '',
                            deviceId: q.get('deviceId') || '',
                            channelId: q.get('channelId') || '0',
                            kitToken: q.get('kitToken') || '',
                            code: q.get('code') || '',
                            streamId: q.get('streamId') || '1',
                            type: '1',
                            recordType: 'cloud',
                        });
                    }""")
            else:
                page.click("#start")
            deadline = time.monotonic() + max(5.0, args.wait_s)
            live = False
            while time.monotonic() < deadline:
                try:
                    info = page.evaluate(
                        """() => ({
                            videos: document.querySelectorAll('video').length,
                            canvas: document.querySelectorAll('canvas').length,
                            root0kids: (document.getElementById('root-0')
                                || {}).querySelectorAll
                                ? document.getElementById('root-0')
                                    .querySelectorAll('*').length : -1,
                        })""")
                except Exception:  # noqa: BLE001
                    info = {}
                if (info.get("videos", 0) + info.get("canvas", 0)) > 0:
                    time.sleep(5.0)
                    try:
                        playing = page.evaluate(
                            """() => Array.from(
                                document.querySelectorAll('video')
                            ).map(v => ({
                                w: v.videoWidth, h: v.videoHeight,
                                t: v.currentTime,
                            }))""")
                    except Exception:  # noqa: BLE001
                        playing = []
                    print(f"video elements: {playing}")
                    live = any(
                        (v.get("w", 0) or 0) > 0 for v in playing)
                    print(f"DEMO LIVE: {'OK' if live else 'FAIL (có khung hình nhưng không có dữ liệu)'}")
                    print("---- console ----")
                    for line in logs[:40]:
                        print(line)
                    browser.close()
                    return
                time.sleep(1.0)
            browser.close()
            print("DEMO LIVE: FAIL (không thấy video/canvas nào)")
    finally:
        if args.demo_via_bridge:
            bridge.close()
        else:
            server.shutdown()
            server.server_close()
    print("---- console ----")
    for line in logs[:40]:
        print(line)


if __name__ == "__main__":
    main()
