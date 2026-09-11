"""Trạng thái nhân viên và sự kiện chuyển trạng thái (định nghĩa bài toán)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WorkState(str, Enum):
    WORKING = "WORKING"  # đang ngồi tại vị trí (detect trong ROI bàn, channel A)
    AWAY_SHORT = "AWAY_SHORT"  # vừa rời vị trí, trong thời gian ân hạn chờ xác nhận
    RESTROOM = "RESTROOM"  # thấy lại ở channel B (hành lang) trong cửa sổ thời gian
    OUT_OF_OFFICE = "OUT_OF_OFFICE"  # quá timeout hoặc qua vùng cửa ra vào
    RETURNED = "RETURNED"  # quay lại vị trí (trạng thái thoáng qua -> WORKING)


@dataclass(slots=True)
class StateEvent:
    timestamp_s: float
    seat_id: str
    prev: WorkState
    new: WorkState
    reason: str
    score: float | None = None
