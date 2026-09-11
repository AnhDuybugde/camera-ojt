"""Business trạng thái theo channel, tách rời khỏi tracking/identity.

Quy ước phân tầng của hệ thống:
- tracking + :class:`GlobalIdentityManager <camera_tracking.tracking.global_identity.GlobalIdentityManager>`
  trả lời "đây là người nào" (Global Person ID ổn định xuyên tracklet,
  xuyên mất dấu, xuyên channel).
- Module này trả lời "người đó đang làm gì ở channel này"
  (làm việc / tạm rời chỗ / có khả năng đi ra ngoài / đang quay lại).

Module này chỉ đọc (global_id, presence, thời gian). Nó KHÔNG bao giờ
ghi ngược vào identity manager, nên logic nghiệp vụ có sai cũng không
thể làm lệch ID. Quy tắc cụ thể của từng channel (ví dụ ROI ghế
channel 1) có thể cắm thêm sau qua `presence_fn` mà không đụng tới ID.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PersonBusinessState(str, Enum):
    """Nghiệp vụ của một người trong một channel cụ thể."""

    UNKNOWN = "UNKNOWN"  # chưa đủ quan sát để kết luận
    WORKING = "WORKING"  # đang hiện diện / ngồi làm việc ở channel này
    AWAY_TEMP = "AWAY_TEMP"  # tạm rời chỗ (vượt grace ngắn)
    POSSIBLY_OUT = "POSSIBLY_OUT"  # vắng lâu, có khả năng đã đi ra ngoài
    RETURNING = "RETURNING"  # vừa xuất hiện trở lại (trạng thái quá độ)


@dataclass
class ChannelBusinessTracker:
    """Theo dõi nghiệp vụ per-Global-ID cho MỘT channel.

    Parameters
    ----------
    channel: tên channel ("A" / "B"), chỉ dùng để hiển thị/log.
    away_grace_s: vắng quá N giây mới chuyển WORKING -> AWAY_TEMP.
    out_after_s: vắng quá N giây thì AWAY_TEMP -> POSSIBLY_OUT.
    return_stable_s: hiện diện trở lại ổn định N giây thì
        RETURNING -> WORKING.
    """

    channel: str = "A"
    away_grace_s: float = 10.0
    out_after_s: float = 60.0
    return_stable_s: float = 3.0
    _last_present_s: dict[int, float] = field(default_factory=dict, init=False)
    _last_absent_s: dict[int, float] = field(default_factory=dict, init=False)
    _returned_at_s: dict[int, float] = field(default_factory=dict, init=False)
    _states: dict[int, PersonBusinessState] = field(default_factory=dict, init=False)

    def update(self, now_s: float, present_gids: set[int]) -> dict[int, PersonBusinessState]:
        """Cập nhật từ tập Global ID đang thấy ở channel này trong frame."""
        for gid in present_gids:
            self._last_present_s[gid] = now_s
            state = self._states.get(gid, PersonBusinessState.UNKNOWN)
            returning_stable = (
                state is PersonBusinessState.RETURNING
                and now_s - self._returned_at_s.get(gid, now_s) >= self.return_stable_s
            )
            if state is PersonBusinessState.UNKNOWN or returning_stable:
                self._states[gid] = PersonBusinessState.WORKING
            elif state in (PersonBusinessState.AWAY_TEMP, PersonBusinessState.POSSIBLY_OUT):
                # Vang mat mot thoi gian roi xuat hien tro lai.
                self._states[gid] = PersonBusinessState.RETURNING
                self._returned_at_s[gid] = now_s
            self._last_absent_s.pop(gid, None)

        for gid, state in list(self._states.items()):
            if gid in present_gids:
                continue
            absent_since = self._last_absent_s.setdefault(
                gid, self._last_present_s.get(gid, now_s)
            )
            missing = now_s - absent_since
            if (
                state in (PersonBusinessState.WORKING, PersonBusinessState.RETURNING)
                and missing >= self.away_grace_s
            ):
                self._states[gid] = PersonBusinessState.AWAY_TEMP
            elif (
                state is PersonBusinessState.AWAY_TEMP and missing >= self.out_after_s
            ):
                self._states[gid] = PersonBusinessState.POSSIBLY_OUT
            # POSSIBLY_OUT + vắng tiếp: giữ nguyên, chờ quay lại.
        return dict(self._states)

    def state_of(self, global_id: int) -> PersonBusinessState:
        return self._states.get(global_id, PersonBusinessState.UNKNOWN)


__all__ = ["ChannelBusinessTracker", "PersonBusinessState"]
