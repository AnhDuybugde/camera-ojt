"""Workstate modules: seat ROI, Re-ID, and cross-camera state machine (MVP)."""
from camera_tracking.workstate.engine import (
    SeatStatus,
    StateEvent,
    WorkStateConfig,
    WorkStateEngine,
)
from camera_tracking.workstate.reid import HistogramEmbedding, cosine_similarity
from camera_tracking.workstate.roi import (
    CorridorZones,
    SeatZone,
    bbox_center_in_polygon,
    point_in_polygon,
    select_seat_occupant,
)
from camera_tracking.workstate.states import WorkState

__all__ = [
    "CorridorZones",
    "HistogramEmbedding",
    "SeatStatus",
    "SeatZone",
    "StateEvent",
    "WorkState",
    "WorkStateConfig",
    "WorkStateEngine",
    "bbox_center_in_polygon",
    "cosine_similarity",
    "point_in_polygon",
    "select_seat_occupant",
]
