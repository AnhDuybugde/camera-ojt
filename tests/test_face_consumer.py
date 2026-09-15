import numpy as np

from camera_tracking.domain import BoundingBox, Frame, Track, TrackEvent
from camera_tracking.face.consumer import FaceTrackConsumer
from camera_tracking.face.embeddings import FaceDetection
from camera_tracking.face.gallery import EnrolledPerson
from camera_tracking.face.matcher import MatchResult


class FakeEmbedder:
    def detect_embed(self, crop_bgr):
        return [FaceDetection((20, 20, 80, 100), 0.99, np.ones(4, dtype=np.float32))]


class FakeMatcher:
    def match(self, embedding):
        person = EnrolledPerson("employee_1", "Employee One", embedding)
        return MatchResult(person, 0.9, True)


def _image() -> np.ndarray:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    checker = np.indices((100, 60)).sum(axis=0) % 2 * 255
    image[40:140, 60:120] = checker[:, :, None]
    return image


def _track() -> Track:
    return Track(7, BoundingBox(40, 20, 160, 180), 0.9, 3, 3, True,
                 local_track_id=12, global_person_id=7)


def test_face_consumer_applies_track_cooldown() -> None:
    image = _image()
    event = TrackEvent("B", Frame(1, 0.0, image), (_track(),))
    consumer = FaceTrackConsumer(
        FakeEmbedder(), FakeMatcher(), min_person_area_px=1,
        min_face_px=20, min_blur_variance=1,
    )

    first = consumer.consume(event, now_s=0.0)
    second = consumer.consume(event, now_s=1.0)
    third = consumer.consume(event, now_s=31.0)

    assert len(first) == 1
    assert second == []
    assert len(third) == 1


def _event(channel: str) -> TrackEvent:
    return TrackEvent(channel, Frame(1, 0.0, _image()), (_track(),))


def test_face_consumer_runs_on_both_channels_by_default() -> None:
    consumer = FaceTrackConsumer(
        FakeEmbedder(), FakeMatcher(), min_person_area_px=1,
        min_face_px=20, min_blur_variance=1,
    )
    assert len(consumer.consume(_event("A"), now_s=0.0)) == 1
    # Gates are per (channel, gid): B is not starved by A's cooldown.
    assert len(consumer.consume(_event("B"), now_s=1.0)) == 1


def test_face_consumer_channel_allowlist() -> None:
    consumer = FaceTrackConsumer(
        FakeEmbedder(), FakeMatcher(), min_person_area_px=1,
        min_face_px=20, min_blur_variance=1, channels=("B",),
    )
    assert consumer.consume(_event("A"), now_s=0.0) == []
    assert len(consumer.consume(_event("B"), now_s=0.0)) == 1


def test_face_consumer_processes_channel_a_independently_from_b() -> None:
    image = _image()
    track = _track()
    consumer = FaceTrackConsumer(
        FakeEmbedder(), FakeMatcher(), min_person_area_px=1,
        min_face_px=20, min_blur_variance=1,
    )

    result_b = consumer.consume(
        TrackEvent("B", Frame(1, 0.0, image), (track,)), now_s=0.0
    )
    result_a = consumer.consume(
        TrackEvent("A", Frame(1, 0.0, image), (track,)), now_s=0.0
    )

    assert len(result_b) == 1
    assert len(result_a) == 1


def test_face_consumer_requires_consensus_before_emitting() -> None:
    consumer = FaceTrackConsumer(
        FakeEmbedder(), FakeMatcher(), min_person_area_px=1,
        min_face_px=20, min_blur_variance=1,
        unknown_cooldown_s=0, consensus_hits=3, consensus_window_s=3,
    )
    event = _event("A")
    assert consumer.consume(event, now_s=0.0) == []
    assert consumer.consume(event, now_s=1.0) == []
    assert len(consumer.consume(event, now_s=2.0)) == 1


def test_one_face_is_assigned_to_only_one_overlapping_track() -> None:
    first = _track()
    second = Track(8, BoundingBox(20, 10, 180, 190), 0.9, 3, 3, True,
                   local_track_id=13, global_person_id=8)
    consumer = FaceTrackConsumer(
        FakeEmbedder(), FakeMatcher(), min_person_area_px=1,
        min_face_px=20, min_blur_variance=1,
    )
    event = TrackEvent("A", Frame(1, 0.0, _image()), (first, second))
    assert len(consumer.consume(event, now_s=0.0)) == 1


class LowScoreEmbedder:
    def detect_embed(self, crop_bgr):
        return [FaceDetection((20, 20, 80, 100), 0.2, np.ones(4, dtype=np.float32))]


def test_face_consumer_filters_low_detector_score() -> None:
    consumer = FaceTrackConsumer(
        LowScoreEmbedder(), FakeMatcher(), min_person_area_px=1,
        min_face_px=20, min_blur_variance=1, min_face_score=0.5,
    )
    assert consumer.consume(_event("A"), now_s=0.0) == []
