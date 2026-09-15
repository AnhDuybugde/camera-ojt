"""Pure helpers for room calibration (no cv2 windows, fully testable).

Two ways to get floor-meter coordinates without a tape measure:
- tiles: offices usually have square floor tiles (e.g. 0.6 m). Click tile
  grid intersections and count (col, row); meters = grid * tile_size.
- paced/rough: walk it off; ROIs only need to be roughly right because
  grace/dwell/hysteresis absorb several decimeters of error.

Fully manual clicking is the fallback. When the floor has visible tiles,
detect_tile_grid() finds the intersections automatically: the only human
input left is ONE number (tile size). Absolute origin does not matter as
long as desks and feet go through the same homography.
"""
from __future__ import annotations

import math


def tile_to_meters(col: int, row: int, tile_m: float) -> tuple[float, float]:
    """Tile grid intersection -> floor meters."""
    if tile_m <= 0:
        raise ValueError("tile_m must be positive")
    return (float(col) * tile_m, float(row) * tile_m)


def validate_polygon(points: list[tuple[float, float]], what: str) -> None:
    if len(points) < 3:
        raise ValueError(f"{what} needs at least 3 points, got {len(points)}")


def validate_workstation(name: str, core: list, extended: list) -> None:
    if not name or not name.strip():
        raise ValueError("workstation needs a non-empty name")
    validate_polygon(core, f"core of {name!r}")
    validate_polygon(extended, f"extended of {name!r}")


def build_location_config(
    base: dict,
    image_points: list[list[float]],
    floor_points: list[list[float]],
    workstations: list[dict],
) -> dict:
    """Return a full config dict with calibration + workstations replaced."""
    import copy

    if len(image_points) < 4 or len(image_points) != len(floor_points):
        raise ValueError("need >= 4 matching calibration point pairs")
    for ws in workstations:
        validate_workstation(ws.get("name", ""), ws.get("core", []),
                             ws.get("extended", []))
    config = copy.deepcopy(base)
    config.setdefault("analytics", {}).setdefault("calibration", {})[
        "image_points"] = [list(map(float, p)) for p in image_points]
    config["analytics"]["calibration"]["floor_points"] = [
        list(map(float, p)) for p in floor_points]
    config["workstations"] = workstations
    return config


__all__ = [
    "build_location_config",
    "detect_tile_grid",
    "tile_to_meters",
    "topdown_pairs",
    "validate_polygon",
    "validate_workstation",
]


def topdown_pairs(
    p1: tuple[float, float],
    p2: tuple[float, float],
    distance_m: float,
) -> tuple[list[list[float]], list[list[float]]]:
    """Near-nadir camera shortcut: 2 clicks a known distance apart.

    Assumes no roll (image x -> floor x, image y -> floor y). Enough when
    the camera looks almost straight down; otherwise use 4+ points.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    pix = math.hypot(dx, dy)
    if pix < 1e-6:
        raise ValueError("two distinct points are required")
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    scale = distance_m / pix
    image_points = [[float(p1[0]), float(p1[1])],
                    [float(p2[0]), float(p2[1])]]
    floor_points = [[0.0, 0.0], [dx * scale, dy * scale]]
    return image_points, floor_points


def detect_tile_grid(
    gray,
    tile_m: float,
    *,
    min_points: int = 12,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 80,
    min_line_length: int = 60,
    max_line_gap: int = 10,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Find tile-grid intersections in a grayscale frame.

    Returns [((px, py), (floor_x, floor_y)), ...]. Raises ValueError when
    too few reliable points are found (caller falls back to manual).
    Needs numpy + cv2 (already project dependencies).
    """
    import numpy as np

    try:
        import cv2
    except ImportError as error:
        raise ValueError("opencv is required for tile detection") from error
    if tile_m <= 0:
        raise ValueError("tile_m must be positive")

    img = np.asarray(gray)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    segments = cv2.HoughLinesP(
        edges, 1, np.pi / 180, hough_threshold,
        minLineLength=min_line_length, maxLineGap=max_line_gap)
    if segments is None or len(segments) < 4:
        raise ValueError("not enough line segments for a tile grid")

    # Split segments into the two dominant (near-perpendicular) directions.
    groups = _split_directions(segments.reshape(-1, 4))
    if groups is None:
        raise ValueError("no two perpendicular line directions found")
    group_a, group_b = groups

    # Intersect every cross-direction pair inside the frame.
    height, width = img.shape[:2]
    raw: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in group_a:
        for x3, y3, x4, y4 in group_b:
            point = _segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
            if point is None:
                continue
            x, y = point
            if 0 <= x < width and 0 <= y < height:
                raw.append((x, y))
    if len(raw) < min_points:
        raise ValueError(f"only {len(raw)} intersections found")
    points = _dedup_points(raw)

    # Tile axes via PCA, spacing via nearest-neighbour peak, indices by
    # quantization. Residual filter drops chandeliers of false hits.
    indexed = _index_grid(points)
    if len(indexed) < min_points:
        raise ValueError(f"only {len(indexed)} regular grid points found")
    return [((float(x), float(y)), (col * tile_m, row * tile_m))
            for (x, y), (col, row) in indexed]


def _segment_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    import math as _math

    return _math.degrees(_math.atan2(y2 - y1, x2 - x1)) % 180.0


def _split_directions(segments):
    """Cluster segments into two groups ~90 degrees apart (or None)."""
    import numpy as np

    angles = np.array([_segment_angle(*s) for s in segments])
    # Circular distance on the 0..180 dial.
    best: tuple[float, float] | None = None
    best_score = -1
    for seed in range(0, 180, 5):
        dist = np.abs((angles - seed + 90) % 180 - 90)
        near = dist < 12.0
        across = np.abs(dist - 90.0) < 12.0
        score = int(near.sum()) + int(across.sum())
        if score > best_score:
            best_score = score
            best = (seed, score)
    if best is None:
        return None
    seed = best[0]
    dist = np.abs((angles - seed + 90) % 180 - 90)
    group_a = segments[dist < 12.0]
    group_b = segments[np.abs(dist - 90.0) < 12.0]
    if len(group_a) < 2 or len(group_b) < 2:
        return None
    return group_a, group_b


def _segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4)
          - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4)
          - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def _dedup_points(points: list[tuple[float, float]],
                  radius: float = 10.0) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    for x, y in points:
        best = None
        for i, (kx, ky) in enumerate(kept):
            if math.hypot(x - kx, y - ky) < radius:
                best = i
                break
        if best is None:
            kept.append((x, y))
        else:
            kx, ky = kept[best]  # average duplicates for a stabler center
            kept[best] = ((kx + x) / 2.0, (ky + y) / 2.0)
    return kept


def _index_grid(points: list[tuple[float, float]]):
    """Assign (col, row) indices to grid points; drop irregular ones."""
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 4:
        return []
    centered = pts - pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt[:2]  # two dominant tile directions
    proj = centered @ axes.T
    spacings = []
    for axis in range(2):
        coords = np.sort(proj[:, axis])
        gaps = np.diff(coords)
        gaps = gaps[gaps > 1e-6]
        if len(gaps):
            spacings.append(float(np.median(gaps)))
    if len(spacings) < 2 or min(spacings) < 3:
        return []
    # Origin is arbitrary (absolute floor origin does not matter as long as
    # desks and feet share the transform): anchor indices at each axis min.
    indexed = []
    for (x, y), (u, v) in zip(points, proj):
        col = round((u - proj[:, 0].min()) / spacings[0])
        row = round((v - proj[:, 1].min()) / spacings[1])
        residual = (abs(u - (proj[:, 0].min() + col * spacings[0]))
                    + abs(v - (proj[:, 1].min() + row * spacings[1])))
        if residual < 0.35 * (spacings[0] + spacings[1]) / 2:
            indexed.append(((x, y), (col, row)))
    return indexed
