"""Tao san WAV chao bang ZeroTTS (1 cau = 1 file, phat 1 session -> khong ngat).

Doc gallery data/images (+ registry.json / face.name_map) roi sinh:
- output/voice_greetings/unknown.wav        <- voice.unknown_phrase
- output/voice_greetings/<person_id>.wav    <- "Xin chào <display_name>"
- output/voice_greetings/manifest.json      <- phrase -> file (pipeline doc)

Lan dau can mang (tai weights ~900MB, cache duoi HF_HOME). Cac lan sau
offline hoan toan. Chay lai khi them/sua nhan vien hoac doi cau chao.

Vi du:
    python scripts/build_greeting_wavs.py --list-voices
    python scripts/build_greeting_wavs.py --voice maichi
    python scripts/build_greeting_wavs.py --force --only QuocNgoc,unknown
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from camera_tracking.config import load_config
from camera_tracking.face.gallery import load_gallery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--voice", default=None, help="Ghi de voice.zerotts_voice.")
    parser.add_argument("--model", default=None, help="Ghi de voice.zerotts_model.")
    parser.add_argument("--out-dir", default=None, help="Ghi de voice.greeting_dir.")
    parser.add_argument("--force", action="store_true", help="Sinh lai ca file da co.")
    parser.add_argument(
        "--only",
        default=None,
        help="Chi sinh cac key nay (cach nhau boi dau phay), vi du: unknown,QuocNgoc.",
    )
    parser.add_argument("--list-voices", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if reconfigure_stdout is not None:
        try:
            reconfigure_stdout(encoding="utf-8")
        except (OSError, ValueError):
            pass
    config = load_config(args.config)
    voice_cfg = config.voice
    model = args.model or voice_cfg.zerotts_model
    voice = args.voice or voice_cfg.zerotts_voice
    out_dir = Path(args.out_dir or voice_cfg.greeting_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from camera_tracking.voice.zerotts_tts import ZeroTTSBackend

    backend = ZeroTTSBackend(model=model, voice=voice)
    if args.list_voices:
        print("Voices:", ", ".join(backend.list_voices()))
        return

    gallery = load_gallery(
        voice_cfg_gallery_dir(config),
        None,
        config.face.name_map,
        config.face.employee_map,
    )
    targets: dict[str, str] = {"unknown": voice_cfg.unknown_phrase}
    for person in gallery.people:
        targets[person.person_id] = f"Xin chào {person.display_name}"
    if args.only:
        wanted = {key.strip() for key in args.only.split(",") if key.strip()}
        unknown_keys = {key for key in wanted if key not in targets}
        if unknown_keys:
            raise SystemExit(f"Key khong co trong gallery: {sorted(unknown_keys)}")
        targets = {key: targets[key] for key in targets if key in wanted}

    manifest_path = out_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            manifest = {}
    manifest.setdefault("phrases", {})
    manifest["model"] = model
    manifest["voice"] = voice

    for key, phrase in sorted(targets.items()):
        dest = out_dir / f"{key}.wav"
        if dest.is_file() and dest.stat().st_size > 0 and not args.force:
            print(f"Giu {dest.name} ({phrase})")
        else:
            backend.save_wav(phrase, dest)
            print(f"Sinh {dest.name} <- {phrase}")
        manifest["phrases"][phrase] = dest.name
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Manifest: {manifest_path} ({len(manifest['phrases'])} cau)")


def voice_cfg_gallery_dir(config) -> Path:  # type: ignore[no-untyped-def]
    return Path(config.face.gallery_dir)


if __name__ == "__main__":
    main()
