"""Image-space zone rules (R1/R2/R3) from data/processed VIA annotations.

Coordinate frame: normalized image 0..1 (x/W, y/H) so the same polygon
works regardless of runtime resize (both channels are resized to
config.camera.frame_width/height in run_workstate*.py).

- Channel A: R1 = away strip near the door (blue in channel_A_preview.jpg).
  center in R1 -> AWAY, else WORKING.
- Channel B: R2 = small green area behind the glass (out-of-door),
  R3 = large yellow door area (at-door).
  center in R2 -> OUT_OF_DOOR, elif center in R3 -> AT_DOOR, else AWAY.

"Touch" = bbox CENTER inside polygon (per product decision 2026-09-19),
not any-overlap and not foot-point. Stabilization (grace/dwell) lives in
ChannelBusinessTracker, not here: these helpers are pure per-frame.
"""
from __future__ import annotations

import json
from pathlib import Path

Point = tuple[float, float]
Polygon = list[Point]

# Preview sizes the VIA polygons were drawn on.
CHANNEL_A_PREVIEW_W = 1920.0
CHANNEL_A_PREVIEW_H = 1080.0
CHANNEL_B_PREVIEW_W = 2880.0
CHANNEL_B_PREVIEW_H = 1620.0

# Defaults converted from data/processed/*.json (normalized 0..1).
# Regenerate: see ticket notes 2026-09-19.
DEFAULT_R1: Polygon = [
    (0.885417, 0.690741),
    (0.828646, 0.894444),
    (0.758854, 0.882407),
    (0.554688, 0.998148),
    (0.930208, 0.996296),
    (0.960938, 0.694444),
    (0.885417, 0.691667),
]

# R2: small green out-of-door area (channel B, 2880x1620).
DEFAULT_R2: Polygon = [
    (0.075, 0.995062),
    (0.065625, 0.748148),
    (0.274306, 0.741975),
    (0.273264, 0.872222),
    (0.303125, 0.908025),
    (0.303819, 0.932716),
    (0.359722, 0.998148),
]

# R3: large yellow at-door area (channel B, 2880x1620).
DEFAULT_R3: Polygon = [
    (0.349306, 0.379012),
    (0.850347, 0.391358),
    (0.997917, 0.991975),
    (0.364236, 0.996296),
]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Even-odd ray casting. Boundary counts as inside."""
    if len(polygon) < 3:
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Boundary check: point on segment (with epsilon).
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) + abs(dy) > 0:
            t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
            if 0.0 <= t <= 1.0:
                px = x1 + t * dx
                py = y1 + t * dy
                if abs(px - x) < 1e-9 and abs(py - y) < 1e-9:
                    return True
        # Ray to +x: count upward/downward crossings.
        if (y1 > y) != (y2 > y):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if xinters >= x:
                inside = not inside
    return inside


def _normalize_polygon(
    xs: list[float], ys: list[float], img_w: float, img_h: float
) -> Polygon:
    return [(float(x) / img_w, float(y) / img_h) for x, y in zip(xs, ys)]


def load_via_polygons(
    path: str | Path, img_w: float, img_h: float
) -> list[Polygon]:
    """Load ALL polygon regions from a VIA JSON, normalized to 0..1."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = data.get("_via_img_metadata", {})
    polys: list[Polygon] = []
    for entry in meta.values():
        for region in entry.get("regions", []):
            shape = region.get("shape_attributes", {})
            if shape.get("name") != "polygon":
                continue
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if len(xs) >= 3 and len(xs) == len(ys):
                polys.append(_normalize_polygon(xs, ys, img_w, img_h))
    return polys


def _polygon_area(poly: Polygon) -> float:
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def load_channel_zones(
    processed_dir: str | Path | None = None,
) -> dict[str, Polygon]:
    """Load {R1, R2, R3} normalized polygons with file fallback to defaults.

    - R1: channel_a_away_zone.json (single polygon, 1920x1080).
    - R2/R3: channel_b_out_door_&door.json, channel_B entry (2880x1620).
      The two B polygons are disambiguated by AREA (smaller=R2 out,
      larger=R3 door) so VIA region order never matters.
    """
    zones: dict[str, Polygon] = {
        "R1": list(DEFAULT_R1),
        "R2": list(DEFAULT_R2),
        "R3": list(DEFAULT_R3),
    }
    if processed_dir is None:
        processed_dir = (
            Path(__file__).resolve().parents[4] / "data" / "processed"
        )
    base = Path(processed_dir)
    try:
        a_path = base / "channel_a_away_zone.json"
        if a_path.is_file():
            polys = load_via_polygons(
                a_path, CHANNEL_A_PREVIEW_W, CHANNEL_A_PREVIEW_H
            )
            if polys:
                # R1 file holds exactly one polygon; take the largest.
                zones["R1"] = max(polys, key=_polygon_area)
    except (OSError, ValueError, KeyError):
        pass
    try:
        candidates = [
            base / "channel_b_out_door_&door.json",
            base / "channel_b_out_door_door.json",
        ]
        b_path = next((p for p in candidates if p.is_file()), None)
        if b_path is not None:
            data = json.loads(b_path.read_text(encoding="utf-8"))
            meta = data.get("_via_img_metadata", {})
            b_polys: list[Polygon] = []
            for _key, entry in meta.items():
                if "channel_B" not in str(entry.get("filename", "")):
                    continue
                for region in entry.get("regions", []):
                    shape = region.get("shape_attributes", {})
                    if shape.get("name") != "polygon":
                        continue
                    xs = shape.get("all_points_x", [])
                    ys = shape.get("all_points_y", [])
                    if len(xs) >= 3 and len(xs) == len(ys):
                        b_polys.append(
                            _normalize_polygon(
                                xs, ys,
                                CHANNEL_B_PREVIEW_W, CHANNEL_B_PREVIEW_H,
                            )
                        )
            if len(b_polys) >= 2:
                ordered = sorted(b_polys, key=_polygon_area)
                zones["R2"] = ordered[0]  # smaller = outside glass
                zones["R3"] = ordered[-1]  # larger = door
            elif len(b_polys) == 1:
                # Single polygon: keep the other default, replace the
                # closer-by-area one.
                only = b_polys[0]
                area = _polygon_area(only)
                if abs(area - _polygon_area(DEFAULT_R2)) < abs(
                    area - _polygon_area(DEFAULT_R3)
                ):
                    zones["R2"] = only
                else:
                    zones["R3"] = only
    except (OSError, ValueError, KeyError):
        pass
    return zones


def classify_channel_a(center_norm: Point, r1: Polygon) -> str:
    """Raw label for channel A: 'AWAY' if center in R1 else 'WORKING'."""
    return "AWAY" if point_in_polygon(center_norm, r1) else "WORKING"


def classify_channel_b(
    center_norm: Point, r2: Polygon, r3: Polygon
) -> str:
    """Raw label for channel B: R2 wins over R3, else AWAY.

    Returns one of 'OUT_OF_DOOR' | 'AT_DOOR' | 'AWAY'.
    """
    if point_in_polygon(center_norm, r2):
        return "OUT_OF_DOOR"
    if point_in_polygon(center_norm, r3):
        return "AT_DOOR"
    return "AWAY"


__all__ = [
    "CHANNEL_A_PREVIEW_W",
    "CHANNEL_A_PREVIEW_H",
    "CHANNEL_B_PREVIEW_W",
    "CHANNEL_B_PREVIEW_H",
    "DEFAULT_R1",
    "DEFAULT_R2",
    "DEFAULT_R3",
    "classify_channel_a",
    "classify_channel_b",
    "load_channel_zones",
    "load_via_polygons",
    "point_in_polygon",
]
