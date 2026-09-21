"""Benchmark detector throughput independently from camera FPS.

Examples:
    python scripts/benchmark_inference.py --device cuda --frames 30
    python scripts/benchmark_inference.py --device cpu --frames 5 --image-size 640

The benchmark uses synthetic frames, so it measures model/runtime overhead and
does not require opening an RTSP stream. Use the reported inference FPS with
``camera.process_every_n_frames`` to choose a sustainable detection rate.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from camera_tracking.detection import YoloPersonDetector, resolve_device


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        try:
            import torch

            torch.cuda.synchronize()
        except ImportError:
            pass


def benchmark(
    model_path: str,
    requested_device: str,
    image_size: int,
    warmup: int,
    frames: int,
    batch_size: int,
) -> dict[str, float | int | str]:
    device = resolve_device(requested_device)
    detector = YoloPersonDetector(
        model_path=model_path,
        image_size=image_size,
        device=device,
    )
    frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    batch = [frame] * batch_size
    for _ in range(warmup):
        detector.detect_batch(batch)
    _sync(device)

    started = time.perf_counter()
    for _ in range(frames):
        detector.detect_batch(batch)
    _sync(device)
    elapsed = time.perf_counter() - started
    batches_per_second = frames / elapsed if elapsed else 0.0
    return {
        "device": device,
        "image_size": image_size,
        "batch_size": batch_size,
        "batches": frames,
        "elapsed_s": round(elapsed, 3),
        "batch_fps": round(batches_per_second, 2),
        "frame_fps": round(batches_per_second * batch_size, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.batch_size < 1:
        parser.error("frames and batch-size must be positive; warmup cannot be negative")
    print(benchmark(
        args.model,
        args.device,
        args.image_size,
        args.warmup,
        args.frames,
        args.batch_size,
    ))


if __name__ == "__main__":
    main()
