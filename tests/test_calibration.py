"""Test calibration helpers: tiles, validation, location config build."""
import pytest

from camera_tracking.calibration import (
    build_location_config,
    detect_tile_grid,
    topdown_pairs,
    tile_to_meters,
    validate_workstation,
)


def test_tile_to_meters() -> None:
    x, y = tile_to_meters(3, 5, 0.6)
    assert (x, y) == pytest.approx((1.8, 3.0))
    with pytest.raises(ValueError):
        tile_to_meters(1, 1, 0.0)


def test_validate_workstation() -> None:
    validate_workstation("desk1", [(0, 0), (1, 0), (1, 1)], [(0, 0), (2, 0), (2, 2)])
    with pytest.raises(ValueError):
        validate_workstation("", [(0, 0), (1, 0), (1, 1)], [(0, 0), (2, 0), (2, 2)])
    with pytest.raises(ValueError):
        validate_workstation("d", [(0, 0), (1, 0)], [(0, 0), (2, 0), (2, 2)])


def test_build_location_config() -> None:
    base = {"camera": {"frame_width": 1280},
            "analytics": {"calibration": {"image_points": [], "floor_points": []}},
            "workstations": []}
    out = build_location_config(
        base,
        [[240, 700], [1040, 700], [820, 260], [460, 260]],
        [[0, 0], [8, 0], [8, 6], [0, 6]],
        [{"name": "desk1", "core": [[2, 2], [3, 2], [3, 3]],
          "extended": [[1, 1], [4, 1], [4, 4]]}],
    )
    assert out["analytics"]["calibration"]["floor_points"][1] == [8.0, 0.0]
    assert out["workstations"][0]["name"] == "desk1"
    assert base["workstations"] == []  # input untouched
    with pytest.raises(ValueError):
        build_location_config(base, [[0, 0]], [[0, 0]], [])


def test_topdown_pairs() -> None:
    img, floor = topdown_pairs((100.0, 200.0), (180.0, 200.0), 0.6)
    assert img == [[100.0, 200.0], [180.0, 200.0]]
    assert floor[0] == [0.0, 0.0]
    assert floor[1][0] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        topdown_pairs((1.0, 1.0), (1.0, 1.0), 0.6)


def _synthetic_tiles(cols: int = 6, rows: int = 5, step: int = 80):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    img = np.full(((rows + 1) * step + 40, (cols + 1) * step + 40), 255,
                  dtype=np.uint8)
    for c in range(cols + 2):
        x = 20 + c * step
        cv2.line(img, (x, 0), (x, img.shape[0]), 0, 3)
    for r in range(rows + 2):
        y = 20 + r * step
        cv2.line(img, (0, y), (img.shape[1], y), 0, 3)
    return img


def test_detect_tile_grid_synthetic() -> None:
    img = _synthetic_tiles()
    found = detect_tile_grid(img, 0.6, min_points=12)
    assert len(found) >= 20
    # Spacing between neighbouring detected columns ~ 1 tile in floor units.
    xs = sorted({round(fx, 3) for _, (fx, _) in found})
    assert len(xs) >= 4


def test_detect_tile_grid_blank_fails() -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError):
        detect_tile_grid(np.full((200, 200), 200, dtype=np.uint8), 0.6)
