"""Appearance embedding + business trạng thái theo channel.

Tầng identity (:mod:`camera_tracking.tracking.global_identity`) trả lời
"đây là người nào". Module này chứa các thành phần phụ trợ không
được phép ảnh hưởng tới ID: embedding ngoại hình và business state.
"""
from camera_tracking.workstate.channel_status import (
    ChannelBusinessTracker,
    PersonBusinessState,
)
from camera_tracking.workstate.reid import HistogramEmbedding, cosine_similarity

__all__ = [
    "ChannelBusinessTracker",
    "HistogramEmbedding",
    "PersonBusinessState",
    "cosine_similarity",
]
