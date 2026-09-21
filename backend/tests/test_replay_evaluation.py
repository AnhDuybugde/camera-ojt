import csv

from camera_tracking.evaluation import evaluate_replay


def _write(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_replay_metrics_detect_employee_error_and_id_switch(tmp_path) -> None:
    boxes = {"x1": 0, "y1": 0, "x2": 10, "y2": 20}
    truth = tmp_path / "truth.csv"
    predictions = tmp_path / "predictions.csv"
    _write(truth, ["frame", "camera", "truth_id", "employee_id", *boxes], [
        {"frame": 1, "camera": "A", "truth_id": "P1", "employee_id": "E1", **boxes},
        {"frame": 2, "camera": "A", "truth_id": "P1", "employee_id": "E1", **boxes},
    ])
    _write(predictions, ["frame", "camera", "global_id", "employee_id", *boxes], [
        {"frame": 1, "camera": "A", "global_id": "1", "employee_id": "E1", **boxes},
        {"frame": 2, "camera": "A", "global_id": "2", "employee_id": "E2", **boxes},
    ])
    metrics = evaluate_replay(truth, predictions)
    assert metrics.matched_observations == 2
    assert metrics.false_employee_assignments == 1
    assert metrics.id_switches == 1
    assert metrics.promoted_global_ids == 2
