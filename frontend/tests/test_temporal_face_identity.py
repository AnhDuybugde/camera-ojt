from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from camera.live_engine import LiveAttendanceEngine
from camera.live_engine import DetectionView
from face.recognizer import Match
from face.temporal_identity import TemporalIdentityResolver
from ui.live_attendance import _recognition_source


def recognized(employee_id="NV005", score=.72):
    return Match(employee_id, "Nguyễn A", "AI", score)


def unknown(score=.12):
    return Match(None, "UNKNOWN", "", score)


def resolver():
    return TemporalIdentityResolver(
        required_votes=2, confirm_ratio=.67, window_seconds=12, ttl_seconds=15,
    )


def test_single_positive_vote_is_only_verifying():
    decision = resolver().update(7, recognized(), 1.0)
    assert decision.status == "verifying"
    assert decision.match.employee_id == "NV005"


def test_two_consistent_votes_confirm_identity():
    service = resolver()
    service.update(7, recognized(score=.70), 1.0)
    decision = service.update(7, recognized(score=.76), 2.0)
    assert decision.status == "confirmed"
    assert decision.match.employee_id == "NV005"
    assert decision.match.similarity == .73


def test_confirmed_identity_survives_one_missing_face_match():
    service = resolver()
    service.update(7, recognized(), 1.0)
    service.update(7, recognized(), 2.0)
    decision = service.update(7, unknown(), 3.0)
    assert decision.status == "cached"
    assert decision.match.employee_id == "NV005"


def test_identity_cache_expires_and_track_can_be_removed():
    service = TemporalIdentityResolver(
        required_votes=2, confirm_ratio=.67, window_seconds=5, ttl_seconds=3,
    )
    service.update(7, recognized(), 1.0)
    service.update(7, recognized(), 2.0)
    assert service.update(7, unknown(), 6.0).status == "unknown"
    service.remove(7)
    assert service.update(7, unknown(), 7.0).status == "unknown"


def test_one_conflicting_identity_does_not_replace_confirmed_employee():
    service = resolver()
    service.update(7, recognized("NV005"), 1.0)
    service.update(7, recognized("NV005"), 2.0)
    decision = service.update(7, recognized("NV003", .8), 3.0)
    assert decision.status == "cached"
    assert decision.match.employee_id == "NV005"


def test_live_quality_filter_rejects_low_score_small_and_blurry_faces(monkeypatch):
    monkeypatch.setattr(
        "camera.live_engine.settings",
        SimpleNamespace(
            face_detection_threshold=.6, live_min_face_size=48,
            live_blur_threshold=10.0,
        ),
    )
    sharp = np.zeros((200, 200, 3), dtype=np.uint8)
    for y in range(0, 200, 8):
        for x in range(0, 200, 8):
            if (x // 8 + y // 8) % 2:
                sharp[y:y + 8, x:x + 8] = 255
    good = SimpleNamespace(bbox=(50, 50, 130, 130), det_score=.9)
    low = SimpleNamespace(bbox=(50, 50, 130, 130), det_score=.2)
    small = SimpleNamespace(bbox=(50, 50, 80, 80), det_score=.9)
    blurry_frame = cv2.GaussianBlur(sharp, (51, 51), 0)
    assert LiveAttendanceEngine._face_quality_ok(sharp, good)
    assert not LiveAttendanceEngine._face_quality_ok(sharp, low)
    assert not LiveAttendanceEngine._face_quality_ok(sharp, small)
    assert not LiveAttendanceEngine._face_quality_ok(blurry_frame, good)


def test_face_must_belong_to_a_detected_person_when_person_tracks_exist():
    inside = SimpleNamespace(bbox=(120, 120, 180, 180))
    outside = SimpleNamespace(bbox=(500, 120, 560, 180))
    people = [(100, 80, 260, 420)]
    assert LiveAttendanceEngine._face_belongs_to_person(inside, people)
    assert not LiveAttendanceEngine._face_belongs_to_person(outside, people)


def test_attendance_is_recorded_only_after_two_consistent_face_votes():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for y in range(40, 160, 8):
        for x in range(80, 200, 8):
            if (x // 8 + y // 8) % 2:
                frame[y:y + 8, x:x + 8] = 255
    face = SimpleNamespace(
        bbox=(80, 40, 200, 160), det_score=.95,
        embedding=np.ones(512, dtype=np.float32),
    )

    class Detector:
        def detect(self, _frame): return [face]

    class Recognizer:
        def match(self, _embedding): return recognized()

    class Attendance:
        def __init__(self): self.calls = 0
        def record(self, _employee_id):
            self.calls += 1
            return SimpleNamespace(message="CHECK-IN SUCCESS")

    attendance = Attendance()
    engine = LiveAttendanceEngine(SimpleNamespace(), Detector(), Recognizer(), attendance)
    engine.liveness = SimpleNamespace(check=lambda _frame, _face: SimpleNamespace(is_live=True))
    first = engine._recognize(frame)
    second = engine._recognize(frame)
    assert first[0].attendance == "VERIFYING" and first[0].employee_id is None
    assert second[0].employee_id == "NV005"
    assert attendance.calls == 1


def test_rtsp_face_recognition_uses_main_stream_but_webcam_is_unchanged():
    source = "rtsp://example/cam/realmonitor?channel=1&subtype=1"
    assert _recognition_source(source).endswith("subtype=0")
    assert _recognition_source(0) == 0


def test_main_stream_face_box_is_scaled_back_to_live_substream():
    display_camera = SimpleNamespace(
        read=lambda: (True, np.zeros((480, 640, 3), dtype=np.uint8)),
    )
    recognition_camera = SimpleNamespace()
    engine = LiveAttendanceEngine(
        display_camera, SimpleNamespace(), SimpleNamespace(), SimpleNamespace(),
        recognition_camera=recognition_camera,
    )
    source = np.zeros((1620, 2880, 3), dtype=np.uint8)
    item = DetectionView((288, 162, 576, 324), "NV005", "A", .7, "VERIFYING")
    scaled = engine._scale_for_display([item], source)
    assert scaled[0].box == (64, 48, 128, 96)
