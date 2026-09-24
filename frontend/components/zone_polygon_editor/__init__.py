"""No-build Streamlit component for drawing normalized camera polygons."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


_COMPONENT = components.declare_component(
    "zone_polygon_editor",
    path=str(Path(__file__).resolve().parent),
)


def zone_polygon_editor(
    *,
    image_data_url: str,
    zones: list[dict[str, Any]],
    camera: str,
    key: str,
) -> dict[str, Any] | None:
    """Render the editor and return its latest validated-save request."""
    return _COMPONENT(
        image=image_data_url,
        zones=zones,
        camera=camera,
        key=key,
        default=None,
    )


__all__ = ["zone_polygon_editor"]
