"""Appearance embedding + business trạng thái theo channel.

Tầng identity (:mod:`camera_tracking.tracking.global_identity`) trả lời
"đây là người nào". Module này chứa các thành phần phụ trợ không
được phép ảnh hưởng tới ID: embedding ngoại hình và business state.
"""
from camera_tracking.workstate.channel_status import (
    ChannelBusinessTracker,
    PersonBusinessState,
)
from camera_tracking.workstate.reconcile import IdentityReconciler
from camera_tracking.workstate.reid import HistogramEmbedding, cosine_similarity
from camera_tracking.workstate.stabilizer import StateStabilizer, WorkstationState
from camera_tracking.workstate.room_fusion import (
    LABEL_AWAY_SEAT,
    LABEL_NEAR_SEAT,
    LABEL_OUT_OFFICE,
    LABEL_RETURNING,
    LABEL_UNKNOWN,
    LABEL_WORKING,
    RoomPersonStatus,
    RoomPresenceAggregator,
)
from camera_tracking.workstate.workstation import (
    WorkstationAssigner,
    WorkstationZone,
    ZoneObs,
)

__all__ = [
    "ChannelBusinessTracker",
    "HistogramEmbedding",
    "IdentityReconciler",
    "LABEL_AWAY_SEAT",
    "LABEL_NEAR_SEAT",
    "LABEL_OUT_OFFICE",
    "LABEL_RETURNING",
    "LABEL_UNKNOWN",
    "LABEL_WORKING",
    "PersonBusinessState",
    "RoomPersonStatus",
    "RoomPresenceAggregator",
    "StateStabilizer",
    "WorkstationAssigner",
    "WorkstationState",
    "WorkstationZone",
    "ZoneObs",
    "cosine_similarity",
]
