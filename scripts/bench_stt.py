"""Benchmark STT CPU tren audio mic that: faster-whisper vs Zipformer.

    PYTHONPATH=src /home/jloy/venvs/cv/bin/python scripts/bench_stt.py [--models small,medium,zipformer]

So sanh latency + transcript de nghe-kiem bang tai. Ket qua -> output/stt_bench.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DUMP_DIR = ROOT / "output" / "voice_dumps"
# File uu tien: co tieng noi that (rms cao), dai vai giay.
PREFERRED = [
    "mic_voice_20260917-202751.wav",
    "mic_test_20260917-202704.wav",
]


def pick_files(limit: int = 6) -> list[Path]:
    files = [DUMP_DIR / name for name in PREFERRED
             if (DUMP_DIR / name).is_file()]
    segs = sorted(DUMP_DIR.glob("seg_*_rms0.0[123]*.wav"),
                  key=lambda p: p.stat().st_size, reverse=True)
    for seg in segs:
        if seg not in files:
            files.append(seg)
        if len(files) >= limit:
            break
    return files[:limit]


def bench_whisper(model_name: str, files: list[Path]) -> dict:
    from faster_whisper import WhisperModel

    from camera_tracking.voice.rtsp_voice_listener import (
        TARGET_RATE,
        transcribe_segment_fw,
    )

    t0 = time.monotonic()
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    load_s = time.monotonic() - t0
    rows = []
    for wav in files:
        import soundfile as sf

        samples, rate = sf.read(str(wav), dtype="int16", always_2d=False)
        import numpy as np

        pcm_arr = np.asarray(samples).reshape(-1)
        if int(rate) != TARGET_RATE:  # dumps mic Vonage truong hop le
            continue
        pcm = pcm_arr.tobytes()
        t1 = time.monotonic()
        text = transcribe_segment_fw(model, pcm, "vi")
        rows.append({"file": wav.name, "seconds": len(pcm_arr) / TARGET_RATE,
                     "latency_s": round(time.monotonic() - t1, 2),
                     "text": text})
    return {"load_s": round(load_s, 1), "rows": rows}


def bench_zipformer(files: list[Path]) -> dict:
    from camera_tracking.voice.sherpa_stt import SherpaZipformerSTT

    stt = SherpaZipformerSTT()
    load_s = stt.warmup()
    rows = []
    for wav in files:
        t1 = time.monotonic()
        text = stt.transcribe_wav(wav)
        import soundfile as sf

        info = sf.info(str(wav))
        rows.append({"file": wav.name,
                     "seconds": round(info.duration, 1),
                     "latency_s": round(time.monotonic() - t1, 2),
                     "text": text})
    return {"load_s": round(load_s, 1), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="small,medium,zipformer")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()
    files = pick_files(args.limit)
    print(f"files ({len(files)}): {[f.name for f in files]}", flush=True)
    results: dict = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"--- {name} ---", flush=True)
        try:
            if name == "zipformer":
                results[name] = bench_zipformer(files)
            else:
                results[name] = bench_whisper(name, files)
        except Exception as error:  # noqa: BLE001
            results[name] = {"error": str(error)[:300]}
            print(f"{name} LOI: {error}", flush=True)
            continue
        print(f"load {results[name]['load_s']}s", flush=True)
        for row in results[name]["rows"]:
            print(f"  {row['file']} ({row['seconds']}s audio): "
                  f"{row['latency_s']}s -> {row['text']!r}", flush=True)
    out = ROOT / "output" / "stt_bench.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
