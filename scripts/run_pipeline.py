from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow this thin script to run before the project is installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_tracking.camera import OpenCVFrameSource
from camera_tracking.config import load_config
from camera_tracking.detection import resolve_device
from camera_tracking.pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run camera tracking pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument("--source", help="Override webcam index, video path, or RTSP URL.")
    parser.add_argument("--max-frames", type=int, help="Stop after this many processed frames.")
    parser.add_argument("--display", action="store_true", help="Show the annotated video window.")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu / auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.source is not None:
        config.camera.source = int(args.source) if args.source.isdigit() else args.source
    if args.display:
        config.output.display = True
    if args.device is not None:
        config.detection.device = args.device
    print(f"Device: {resolve_device(config.detection.device)}")

    pipeline = build_pipeline(config)
    with OpenCVFrameSource(
        source=config.camera.source,
        width=config.camera.frame_width,
        height=config.camera.frame_height,
        requested_fps=config.camera.fps,
        process_every_n_frames=config.camera.process_every_n_frames,
    ) as source:
        snapshot = pipeline.run(source, max_frames=args.max_frames)

    if snapshot is None:
        print("No frames were processed.")
        return
    print(
        f"Done: occupancy={snapshot.occupancy}, entries={snapshot.entries}, "
        f"exits={snapshot.exits}, report={config.output.output_dir / config.output.report_filename}"
    )


if __name__ == "__main__":
    main()
