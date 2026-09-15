from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_tracking.evaluation import evaluate_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a two-camera identity trace.")
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--min-iou", type=float, default=0.50)
    args = parser.parse_args()
    metrics = evaluate_replay(args.truth, args.predictions, min_iou=args.min_iou)
    print(json.dumps(metrics.as_dict(), indent=2))
    if metrics.false_employee_assignments or metrics.employee_uniqueness_violations:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
