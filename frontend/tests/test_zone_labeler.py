from io import BytesIO

from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from ui.zone_labeler import (
    _apply_editor_result,
    _empty_config,
    _load_config,
    _normalize_image,
    _save_config,
    _validate_zones,
)
from scripts.migrate_zone_labels import migrate_payload


def _zone(**overrides):
    zone = {
        "id": "be_xinh",
        "label": "Vùng chào Bé Xinh",
        "kind": "be_xinh",
        "points": [[0.1, 0.2], [0.8, 0.2], [0.7, 0.9]],
        "color": "#22c55e",
        "privacy": "standard",
        "enabled": True,
    }
    zone.update(overrides)
    return zone


def test_validate_zone_normalizes_and_preserves_points() -> None:
    zones = _validate_zones([_zone()])

    assert zones[0]["id"] == "be_xinh"
    assert zones[0]["points"] == [[0.1, 0.2], [0.8, 0.2], [0.7, 0.9]]


def test_restroom_always_enforces_anonymous_audio() -> None:
    zones = _validate_zones([_zone(id="restroom", kind="restroom")])

    assert zones[0]["privacy"] == "anonymous_audio"


@pytest.mark.parametrize(
    "points",
    [
        [[0.1, 0.1], [0.2, 0.2]],
        [[0.1, 0.1], [1.2, 0.2], [0.2, 0.8]],
        [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]],
    ],
)
def test_invalid_polygon_is_rejected(points) -> None:
    with pytest.raises(ValueError):
        _validate_zones([_zone(points=points)])


def test_config_round_trip(tmp_path) -> None:
    path = tmp_path / "zone_labels.json"
    config = _empty_config()
    config["cameras"]["A"]["zones"] = [_zone()]

    _save_config(config, path)
    loaded = _load_config(path)

    assert loaded["coordinate_space"] == "normalized_image"
    assert loaded["anchor"] == "bbox_bottom_center"
    assert loaded["cameras"]["A"]["zones"][0]["kind"] == "be_xinh"


def test_editor_result_is_saved_immediately(tmp_path) -> None:
    path = tmp_path / "zone_labels.json"
    config = _empty_config()

    zones = _apply_editor_result(
        config,
        "A",
        {"camera": "A", "zones": [_zone()]},
        path,
    )

    assert zones[0]["id"] == "be_xinh"
    assert _load_config(path)["cameras"]["A"]["zones"] == zones


def test_imou_reference_crop_removes_side_bars() -> None:
    source = Image.new("RGB", (1908, 858), "white")
    buffer = BytesIO()
    source.save(buffer, format="JPEG")

    normalized, size = _normalize_image(
        buffer.getvalue(), crop_imou_reference=True
    )

    assert normalized.startswith(b"\xff\xd8")
    assert size == (1526, 858)


def test_zone_labeler_page_smoke() -> None:
    app = AppTest.from_string(
        "from ui.zone_labeler import render\nrender('ADMIN')",
        default_timeout=15,
    ).run()

    assert not app.exception


def test_migrate_imou_padding_and_semantics() -> None:
    payload = _empty_config()
    payload["cameras"]["A"] = {
        "image_size": [1908, 858],
        "zones": [_zone(privacy="anonymous_audio")],
    }
    payload["cameras"]["B"] = {
        "image_size": [1908, 858],
        "zones": [_zone(id="di_vao", label="đi vào", kind="door_outside")],
    }

    migrated, changed = migrate_payload(payload)

    assert changed
    assert migrated["cameras"]["A"]["image_size"] == [1526, 858]
    assert migrated["cameras"]["A"]["zones"][0]["privacy"] == "standard"
    assert migrated["cameras"]["B"]["zones"][0]["kind"] == "door_inside"
    assert migrate_payload(migrated)[1] is False
