"""One-shot smoke test: TTS/WAV -> P2P VisualTalk -> loa camera (khong browser).

Vi du:
    python scripts/test_imou_p2p_voice.py --text "Xin chào Duy"
    python scripts/test_imou_p2p_voice.py --audio audio/camera-test-vi.wav
    python scripts/test_imou_p2p_voice.py --greeting QuocNgoc
    python scripts/test_imou_p2p_voice.py --greeting unknown

Can .env: IMOU_DEVICE_ID (serial) + IMOU_CAMERA_PASSWORD (safety code,
fallback IMOU_PASSWORD / IMOU_DEVICE_CODE). Can mang ra Imou P2P cloud +
ffmpeg (pip install imageio-ffmpeg). Khong mo mic laptop, khong phat loa laptop.
"""
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
from camera_tracking.voice.p2p_talk import ImouP2PCredentials, send_audio_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--text", default="Xin chào, đây là thử loa camera")
    parser.add_argument("--audio", default=None, help="Gui thang file co san, bo qua TTS.")
    parser.add_argument(
        "--greeting",
        default=None,
        help="Gui WAV tao san theo key manifest (person_id hoac 'unknown'), "
        "vi du: --greeting QuocNgoc.",
    )
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--attempts", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    config = load_config(args.config)
    voice = config.voice

    if args.greeting:
        from camera_tracking.voice.zerotts_tts import load_phrase_files

        candidates = load_phrase_files(voice.greeting_dir)
        match = None
        for wav in candidates.values():
            if wav.stem == args.greeting:
                match = wav
                break
        if match is None:
            raise SystemExit(
                f"Greeting '{args.greeting}' chua co; chay "
                f"scripts/build_greeting_wavs.py truoc."
            )
        audio_path = match
        print(f"Pregen greeting: {audio_path}")
    elif args.audio:
        audio_path = Path(args.audio)
        if not audio_path.is_file():
            raise SystemExit(f"Audio file not found: {audio_path}")
    else:
        greeter = VoiceGreeter(cache_dir=voice.cache_dir, voice=voice.voice)
        audio_path = greeter.synthesize(args.text)
        if audio_path is None:
            raise RuntimeError("Không tạo được TTS; kiểm tra edge-tts và kết nối mạng")
        print(f"TTS cached: {audio_path}")

    creds = ImouP2PCredentials.from_env()
    print(f"P2P talk -> serial ...{creds.serial[-4:]} channel={args.channel or voice.p2p_channel}")
    send_audio_file(
        audio_path,
        creds,
        channel=args.channel or voice.p2p_channel,
        timeout=args.timeout or voice.p2p_timeout_s,
        attempts=args.attempts or voice.p2p_attempts,
        retry_delay=voice.p2p_retry_delay_s,
        sample_rate=voice.p2p_sample_rate,
    )
    print("Sent DHAV audio frames to the camera (200 OK + frames = transport OK).")


if __name__ == "__main__":
    main()
