from __future__ import annotations

from unittest import TestCase

import numpy as np

from camera_tracking.domain import BoundingBox, Track
from camera_tracking.workstate import (
    HistogramEmbedding,
    SeatZone,
    WorkState,
    WorkStateConfig,
    WorkStateEngine,
    bbox_center_in_polygon,
    cosine_similarity,
    point_in_polygon,
    select_seat_occupant,
)


def unit_vector(first_nonzero_at: int, dim: int = 8) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[first_nonzero_at] = 1.0
    return vec


class RoiTest(TestCase):
    def test_point_in_polygon(self) -> None:
        square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        self.assertTrue(point_in_polygon((5.0, 5.0), square))
        self.assertFalse(point_in_polygon((15.0, 5.0), square))

    def test_bbox_center(self) -> None:
        square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        self.assertTrue(bbox_center_in_polygon(BoundingBox(4, 4, 6, 6), square))
        self.assertFalse(bbox_center_in_polygon(BoundingBox(20, 20, 30, 30), square))

    def test_selects_person_closest_to_seat_edge(self) -> None:
        seat = SeatZone("seat_01", "Seat 1", [(0, 0), (100, 0), (100, 100), (0, 100)])
        rear = Track(10, BoundingBox(35, 10, 65, 45), 0.9, 1, 2, True)
        occupant = Track(20, BoundingBox(35, 35, 65, 90), 0.9, 1, 2, True)

        selected, ambiguous = select_seat_occupant(seat, [rear, occupant])

        self.assertFalse(ambiguous)
        self.assertEqual(selected, occupant)

    def test_keeps_preferred_person_when_multiple_are_in_roi(self) -> None:
        seat = SeatZone("seat_01", "Seat 1", [(0, 0), (100, 0), (100, 100), (0, 100)])
        preferred = Track(10, BoundingBox(10, 20, 40, 70), 0.9, 1, 2, True)
        closer = Track(20, BoundingBox(35, 35, 65, 90), 0.9, 1, 2, True)

        selected, ambiguous = select_seat_occupant(
            seat,
            [preferred, closer],
            preferred_track_id=10,
        )

        self.assertFalse(ambiguous)
        self.assertEqual(selected, preferred)


class ReidTest(TestCase):
    def test_cosine(self) -> None:
        left = unit_vector(0)
        self.assertAlmostEqual(cosine_similarity(left, left), 1.0)
        self.assertAlmostEqual(cosine_similarity(left, unit_vector(1)), 0.0)

    def test_histogram_match(self) -> None:
        extractor = HistogramEmbedding()
        query = unit_vector(2)
        candidates = [unit_vector(0), unit_vector(2)]
        idx, score = extractor.match(query, candidates, threshold=0.5)
        self.assertEqual(idx, 1)
        self.assertAlmostEqual(score, 1.0)
        idx, _ = extractor.match(query, [unit_vector(0)], threshold=0.5)
        self.assertIsNone(idx)


class EngineTest(TestCase):
    def make_engine(self) -> WorkStateEngine:
        return WorkStateEngine(
            seat_ids=["seat_01"],
            config=WorkStateConfig(
                leave_grace_s=10.0,
                corridor_match_window_s=300.0,
                restroom_return_window_s=900.0,
                similarity_threshold=0.5,
                returned_promote_s=3.0,
            ),
        )

    def test_unobserved_seat_remains_unknown(self) -> None:
        engine = self.make_engine()

        self.assertEqual(engine.state_of("seat_01"), WorkState.UNKNOWN)
        self.assertEqual(engine.update_office(0.0, {}), [])
        self.assertEqual(engine.tick(3600.0), [])
        self.assertEqual(engine.state_of("seat_01"), WorkState.UNKNOWN)

    def test_leave_then_restroom_then_return(self) -> None:
        engine = self.make_engine()
        emb = unit_vector(0)

        # Đang ngồi.
        events = engine.update_office(0.0, {"seat_01": True}, {"seat_01": emb})
        self.assertEqual(events[0].new, WorkState.WORKING)
        # Rời 5s (chưa quá grace 10s) -> chưa chuyển.
        self.assertEqual(engine.update_office(5.0, {"seat_01": False}), [])
        self.assertEqual(engine.state_of("seat_01"), WorkState.WORKING)
        # Vắng liên tục 5 -> 16 (11s) -> AWAY_SHORT.
        events = engine.update_office(16.0, {"seat_01": False})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].new, WorkState.AWAY_SHORT)

        # Xuất hiện ở hành lang với embedding khớp -> RESTROOM.
        events = engine.update_corridor(60.0, [unit_vector(0)], ["hallway"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].new, WorkState.RESTROOM)

        # Quay lại ghế -> RETURNED rồi về WORKING.
        events = engine.update_office(200.0, {"seat_01": True}, {"seat_01": emb})
        self.assertEqual(events[0].new, WorkState.RETURNED)
        events = engine.update_office(204.0, {"seat_01": True}, {"seat_01": emb})
        self.assertEqual(events[0].new, WorkState.WORKING)

    def test_away_timeout_goes_out(self) -> None:
        engine = self.make_engine()
        emb = unit_vector(1)
        engine.update_office(0.0, {"seat_01": True}, {"seat_01": emb})
        engine.update_office(1.0, {"seat_01": False})
        engine.update_office(12.0, {"seat_01": False})
        self.assertEqual(engine.state_of("seat_01"), WorkState.AWAY_SHORT)
        events = engine.tick(500.0)  # quá 300s không thấy ở B
        self.assertEqual(events[0].new, WorkState.OUT_OF_OFFICE)

    def test_exit_door_goes_out_immediately(self) -> None:
        engine = self.make_engine()
        emb = unit_vector(3)
        engine.update_office(0.0, {"seat_01": True}, {"seat_01": emb})
        engine.update_office(1.0, {"seat_01": False})
        engine.update_office(12.0, {"seat_01": False})
        events = engine.update_corridor(30.0, [unit_vector(3)], ["exit"])
        self.assertEqual(events[0].new, WorkState.OUT_OF_OFFICE)

    def test_wrong_person_does_not_match(self) -> None:
        engine = self.make_engine()
        engine.update_office(0.0, {"seat_01": True}, {"seat_01": unit_vector(0)})
        engine.update_office(1.0, {"seat_01": False})
        engine.update_office(12.0, {"seat_01": False})
        events = engine.update_corridor(30.0, [unit_vector(7)], ["hallway"])
        self.assertEqual(events, [])
        self.assertEqual(engine.state_of("seat_01"), WorkState.AWAY_SHORT)
