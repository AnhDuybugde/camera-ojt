from pathlib import Path

from camera_tracking.config import load_config


def test_load_config() -> None:
    config = load_config(Path("config/default.yaml"))

    assert config["camera"]["source"] == 0
    assert config["detection"]["person_class_id"] == 0
