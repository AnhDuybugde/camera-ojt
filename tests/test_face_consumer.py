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


def test_face_consumer_applies_track_cooldown() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    checker = np.indices((100, 60)).sum(axis=0) % 2 * 255
    image[40:140, 60:120] = checker[:, :, None]
    track = Track(7, BoundingBox(40, 20, 160, 180), 0.9, 3, 3, True,
                  local_track_id=12, global_person_id=7)
    event = TrackEvent("B", Frame(1, 0.0, image), (track,))
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
