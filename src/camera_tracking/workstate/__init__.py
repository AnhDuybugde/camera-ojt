"""Appearance embedding + business trạng thái theo channel.

Tầng identity (:mod:`camera_tracking.tracking.global_identity`) trả lời
"đây là người nào". Module này chứa các thành phần phụ trợ không
được phép ảnh hưởng tới ID: embedding ngoại hình và business state.
"""
from camera_tracking.workstate.channel_status import (
    ChannelBusinessTracker,
    PersonBusinessState,
)
from camera_tracking.workstate.consumer import WorkstateConsumer
from camera_tracking.workstate.image_zones import (
    classify_channel_a,
    classify_channel_b,
    load_channel_zones,
    point_in_polygon,
)
from camera_tracking.workstate.reconcile import IdentityReconciler
from camera_tracking.workstate.reid import OsnetEmbedding, cosine_similarity
from backend.app.services.room_status_service import (
    LABEL_AT_DOOR,
    LABEL_AWAY,
    LABEL_OUT_OF_DOOR,
    LABEL_UNKNOWN,
    LABEL_WORKING,
    RoomPersonStatus,
    RoomPresenceAggregator,
)
from camera_tracking.workstate.stabilizer import StateStabilizer, WorkstationState
from camera_tracking.workstate.workstation import (
    WorkstationAssigner,
    WorkstationZone,
    ZoneObs,
)

__all__ = [
    "LABEL_AT_DOOR",
    "LABEL_AWAY",
    "LABEL_OUT_OF_DOOR",
    "LABEL_UNKNOWN",
    "LABEL_WORKING",
    "ChannelBusinessTracker",
    "IdentityReconciler",
    "OsnetEmbedding",
    "PersonBusinessState",
    "RoomPersonStatus",
    "RoomPresenceAggregator",
    "StateStabilizer",
    "WorkstateConsumer",
    "WorkstationAssigner",
    "WorkstationState",
    "WorkstationZone",
    "ZoneObs",
    "classify_channel_a",
    "classify_channel_b",
    "cosine_similarity",
    "load_channel_zones",
    "point_in_polygon",
]
