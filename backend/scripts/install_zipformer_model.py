"""Download the Vietnamese Zipformer INT8 files required by sherpa-onnx.

The model weights are runtime assets and stay outside Git. Re-running this
script is safe: valid existing files are retained and partial downloads use a
temporary suffix before an atomic rename.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from camera_tracking.voice.sherpa_stt import MODEL_DIR_NAME, default_model_dir

REPOSITORY = f"csukuangfj2/{MODEL_DIR_NAME}"
FILES = {
    "tokens.txt": 10_000,
    "encoder.int8.onnx": 20_000_000,
    "decoder.onnx": 4_000_000,
    "joiner.int8.onnx": 500_000,
}


def _download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "camera-ojt/zipformer-installer"})
    try:
        with urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def install(destination: Path, *, force: bool = False) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for filename, minimum_bytes in FILES.items():
        target = destination / filename
        if not force and target.is_file() and target.stat().st_size >= minimum_bytes:
            print(f"[Zipformer] da co {filename} ({target.stat().st_size:,} bytes)")
            continue
        print(f"[Zipformer] dang tai {filename}...", flush=True)
        url = f"https://huggingface.co/{REPOSITORY}/resolve/main/{filename}?download=true"
        _download(url, target)
        if target.stat().st_size < minimum_bytes:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"file tai ve khong hop le: {filename}")
        print(f"[Zipformer] xong {filename} ({target.stat().st_size:,} bytes)")
    print(f"[Zipformer] san sang tai {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=default_model_dir())
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    install(args.destination, force=args.force)


if __name__ == "__main__":
    main()
