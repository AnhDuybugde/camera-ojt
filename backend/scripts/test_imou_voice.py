"""One-shot smoke test: TTS -> Imou WebSDK AudioTalk -> camera speaker."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser.add_argument("--text", default="Xin chào, đây là thử loa camera")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    config = load_config(args.config)
    voice = config.voice
    greeter = VoiceGreeter(cache_dir=voice.cache_dir, voice=voice.voice)
    audio_path = greeter.synthesize(args.text)
    if audio_path is None:
        raise RuntimeError("Không tạo được TTS; kiểm tra edge-tts và kết nối mạng")

    bridge = ImouAudioTalkBridge.from_env(
        host=voice.bridge_host,
        port=voice.bridge_port,
        command_ttl_s=voice.command_ttl_s,
        talk_tail_s=voice.talk_tail_s,
        launch_browser=voice.launch_browser,
    )
    try:
        bridge.start()
        print(f"AudioTalk bridge: {bridge.url.split('?')[0]}")
        try:
            result = bridge.play(audio_path)
        except Exception:
            print(f"Bridge page state: {bridge.debug_state()}")
            raise
        print(f"Camera playback: {result.status}")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
