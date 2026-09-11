"""State machine xuyên camera (MVP).

Luồng:
  WORKING --(rời ROI > leave_grace_s)--> AWAY_SHORT (lưu embedding lúc rời)
  AWAY_SHORT --(khớp ở hành lang B trong corridor_match_window_s)--> RESTROOM
  AWAY_SHORT --(timeout)--> OUT_OF_OFFICE
  RESTROOM --(quay lại ghế)--> RETURNED -> WORKING
  RESTROOM --(quá restroom_return_window_s)--> OUT_OF_OFFICE
  OUT_OF_OFFICE --(quay lại ghế)--> RETURNED -> WORKING
  Bất kỳ AWAY/RESTROOM nào --(khớp ở vùng cửa exit)--> OUT_OF_OFFICE

Engine không phụ thuộc OpenCV/torch để dễ test: nhận vào
presence bool + embedding vector đã trích sẵn.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from camera_tracking.workstate.reid import cosine_similarity
from camera_tracking.workstate.states import StateEvent, WorkState


@dataclass(slots=True)
class WorkStateConfig:
    leave_grace_s: float = 10.0  # rời ROI quá N giây mới tính là rời
    corridor_match_window_s: float = 300.0  # 0-5 phút: A -> B coi là đi vệ sinh
    restroom_return_window_s: float = 900.0  # 15 phút phải quay lại
    similarity_threshold: float = 0.55  # ngưỡng cosine cho histogram MVP
    returned_promote_s: float = 3.0  # RETURNED tự về WORKING sau N giây


@dataclass(slots=True)
class SeatStatus:
    seat_id: str
    state: WorkState = WorkState.UNKNOWN
    last_in_seat_s: float | None = None
    away_candidate_since_s: float | None = None
    away_since_s: float | None = None
    pending_emb: np.ndarray | None = None
    corridor_seen_s: float | None = None
    last_event_s: float = 0.0


@dataclass
class WorkStateEngine:
    seat_ids: list[str]
    config: WorkStateConfig = field(default_factory=WorkStateConfig)
    statuses: dict[str, SeatStatus] = field(init=False)
    events: list[StateEvent] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.statuses = {sid: SeatStatus(seat_id=sid) for sid in self.seat_ids}

    # -- helpers ---------------------------------------------------------
    def _emit(
        self,
        timestamp_s: float,
        seat_id: str,
        prev: WorkState,
        new: WorkState,
        reason: str,
        score: float | None = None,
    ) -> StateEvent:
        event = StateEvent(timestamp_s, seat_id, prev, new, reason, score)
        self.events.append(event)
        status = self.statuses[seat_id]
        status.state = new
        status.last_event_s = timestamp_s
        return event

    # -- channel A -------------------------------------------------------
    def update_office(
        self,
        timestamp_s: float,
        presence: dict[str, bool],
        embeddings: dict[str, np.ndarray | None] | None = None,
    ) -> list[StateEvent]:
        """Gọi mỗi frame (hoặc mỗi N frame) của camera A."""
        embeddings = embeddings or {}
        out: list[StateEvent] = []
        for seat_id, is_present in presence.items():
            if seat_id not in self.statuses:
                continue
            st = self.statuses[seat_id]
            emb = embeddings.get(seat_id)

            if is_present:
                st.last_in_seat_s = timestamp_s
                st.away_candidate_since_s = None
                if st.state == WorkState.UNKNOWN:
                    out.append(
                        self._emit(
                            timestamp_s,
                            seat_id,
                            st.state,
                            WorkState.WORKING,
                            "xác nhận có mặt tại vị trí",
                        )
                    )
                elif st.state == WorkState.RETURNED:
                    # Đủ lâu ở ghế -> về WORKING hẳn.
                    if timestamp_s - st.last_event_s >= self.config.returned_promote_s:
                        out.append(
                            self._emit(timestamp_s, seat_id, st.state,
                                       WorkState.WORKING, "ổn định lại vị trí")
                        )
                elif st.state in (
                    WorkState.AWAY_SHORT,
                    WorkState.RESTROOM,
                    WorkState.OUT_OF_OFFICE,
                ):
                    prev = st.state
                    out.append(
                        self._emit(timestamp_s, seat_id, prev,
                                   WorkState.RETURNED, "quay lại vị trí")
                    )
                    st.away_since_s = None
                    st.corridor_seen_s = None
                    st.pending_emb = None
                # WORKING + present: không làm gì.
                if emb is not None:
                    # Luôn refresh embedding lúc còn ngồi để khi rời có mẫu mới nhất.
                    st.pending_emb = emb
            else:  # vắng mặt ở ghế
                if st.state == WorkState.WORKING:
                    if st.away_candidate_since_s is None:
                        st.away_candidate_since_s = timestamp_s
                    elif timestamp_s - st.away_candidate_since_s >= self.config.leave_grace_s:
                        prev = st.state
                        st.away_since_s = timestamp_s
                        # Giữ embedding mới nhất lúc còn ngồi (đã refresh ở trên).
                        out.append(
                            self._emit(timestamp_s, seat_id, prev,
                                       WorkState.AWAY_SHORT,
                                       f"rời vị trí quá {self.config.leave_grace_s:g}s")
                        )
                        st.away_candidate_since_s = None
                elif st.state == WorkState.RETURNED:
                    # Vừa về lại đi ngay -> quay về AWAY_SHORT.
                    out.append(
                        self._emit(timestamp_s, seat_id, st.state,
                                   WorkState.AWAY_SHORT, "lại rời vị trí")
                    )
                    st.away_since_s = timestamp_s
                # AWAY_SHORT/RESTROOM/OUT + vắng: chờ corridor/tick xử lý.
        return out

    # -- channel B -------------------------------------------------------
    def update_corridor(
        self,
        timestamp_s: float,
        candidate_embs: list[np.ndarray | None],
        candidate_zones: list[str] | None = None,
    ) -> list[StateEvent]:
        """Gọi mỗi khi channel B detect được người.

        candidate_zones: mỗi phần tử là 'hallway' | 'exit' | 'unknown'.
        """
        out: list[StateEvent] = []
        if not candidate_embs:
            return out
        zones = candidate_zones or ["unknown"] * len(candidate_embs)
        used_candidate: set[int] = set()

        # Ưu tiên vùng exit trước: khớp -> OUT_OF_OFFICE ngay.
        for seat_id, st in self.statuses.items():
            if st.state not in (WorkState.AWAY_SHORT, WorkState.RESTROOM):
                continue
            if st.pending_emb is None:
                continue
            if timestamp_s - (st.away_since_s or timestamp_s) > self.config.corridor_match_window_s + self.config.restroom_return_window_s:
                continue
            best_idx, best_score = self._best_match(st.pending_emb, candidate_embs, used_candidate)
            if best_idx is None:
                continue
            if zones[best_idx] == "exit":
                used_candidate.add(best_idx)
                out.append(
                    self._emit(timestamp_s, seat_id, st.state,
                               WorkState.OUT_OF_OFFICE,
                               "đi qua cửa ra vào", best_score)
                )

        # Vùng hallway: AWAY_SHORT -> RESTROOM.
        for seat_id, st in self.statuses.items():
            if st.state != WorkState.AWAY_SHORT:
                continue
            if st.pending_emb is None:
                continue
            if st.away_since_s is None:
                continue
            if timestamp_s - st.away_since_s > self.config.corridor_match_window_s:
                continue  # hết cửa sổ -> để tick() chuyển OUT
            best_idx, best_score = self._best_match(st.pending_emb, candidate_embs, used_candidate)
            if best_idx is None:
                continue
            if zones[best_idx] == "exit":
                continue  # đã xử lý ở vòng exit
            used_candidate.add(best_idx)
            st.corridor_seen_s = timestamp_s
            out.append(
                self._emit(timestamp_s, seat_id, st.state,
                           WorkState.RESTROOM,
                           "xuất hiện ở hành lang trong cửa sổ 5 phút", best_score)
            )
        return out

    def _best_match(
        self,
        query: np.ndarray,
        candidates: list[np.ndarray | None],
        skip: set[int],
    ) -> tuple[int | None, float]:
        best_idx: int | None = None
        best_score = -1.0
        for idx, cand in enumerate(candidates):
            if idx in skip or cand is None:
                continue
            score = cosine_similarity(query, cand)
            if score > best_score:
                best_score = score
                best_idx = idx
        best_score = max(best_score, 0.0)
        if best_idx is None or best_score < self.config.similarity_threshold:
            return None, best_score
        return best_idx, best_score

    def update_corridor_scored(
        self, timestamp_s: float, scored: list[tuple[np.ndarray | None, str, float]]
    ) -> list[StateEvent]:
        """Biến thể khi caller đã tính score sẵn (ít dùng ở MVP)."""
        embs = [s[0] for s in scored]
        zones = [s[1] for s in scored]
        return self.update_corridor(timestamp_s, embs, zones)

    # -- timeouts --------------------------------------------------------
    def tick(self, timestamp_s: float) -> list[StateEvent]:
        out: list[StateEvent] = []
        cfg = self.config
        for seat_id, st in self.statuses.items():
            if st.state == WorkState.AWAY_SHORT and st.away_since_s is not None:
                if timestamp_s - st.away_since_s > cfg.corridor_match_window_s:
                    out.append(
                        self._emit(timestamp_s, seat_id, st.state,
                                   WorkState.OUT_OF_OFFICE,
                                   f"quá {cfg.corridor_match_window_s:g}s không thấy ở hành lang")
                    )
            elif st.state == WorkState.RESTROOM and st.corridor_seen_s is not None:
                if timestamp_s - st.corridor_seen_s > cfg.restroom_return_window_s:
                    out.append(
                        self._emit(timestamp_s, seat_id, st.state,
                                   WorkState.OUT_OF_OFFICE,
                                   f"quá {cfg.restroom_return_window_s:g}s chưa quay lại")
                    )
            elif (
                st.state == WorkState.RETURNED
                and timestamp_s - st.last_event_s >= cfg.returned_promote_s
            ):
                out.append(
                    self._emit(timestamp_s, seat_id, st.state,
                               WorkState.WORKING, "ổn định lại vị trí")
                )
        return out

    def state_of(self, seat_id: str) -> WorkState:
        return self.statuses[seat_id].state
