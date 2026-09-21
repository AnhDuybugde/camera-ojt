"""Build conservative face/body thresholds from a labeled score CSV."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True,
                        help="CSV columns: kind,label,score[,second_score]")
    parser.add_argument("--margin", type=float, default=0.02)
    args = parser.parse_args()
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    impostor_face_margins: list[float] = []
    with args.scores.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["kind"]][row["label"]].append(float(row["score"]))
            if row["kind"] == "face" and row["label"] == "impostor" \
                    and row.get("second_score"):
                impostor_face_margins.append(
                    float(row["score"]) - float(row["second_score"])
                )
    thresholds = {}
    for kind in ("face", "body"):
        impostors = grouped[kind]["impostor"]
        if not impostors:
            raise SystemExit(f"missing {kind} impostor scores")
        floor = 0.70 if kind == "face" else 0.0
        thresholds[f"{kind}_threshold"] = max(
            floor, max(impostors) + args.margin
        )
    thresholds["face_min_margin"] = max(
        0.10,
        (max(impostor_face_margins) + args.margin)
        if impostor_face_margins else 0.10,
    )
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
