from __future__ import annotations

import argparse
from pathlib import Path

from camera_tracking.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run camera tracking pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Camera source: {config['camera']['source']}")


if __name__ == "__main__":
    main()
