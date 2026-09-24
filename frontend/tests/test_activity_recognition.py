from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np

from camera.live_engine import DetectionView, LiveAttendanceEngine
from spatial.activity_config import ActivityConfig
from spatial.activity_engine import ActivityRuntime, ActivityView, IdentityObservation
from spatial.activity_rules import (
    ActivityType, RuleResult, TemporalActivityState, classify_activity,
    smooth_activity,
)
from spatial.pose_service import PoseObservation
from spatial.repository import SpatialRepository
from spatial.tracker import TrackSnapshot
from spatial.zones import Zone
from ui.live_attendance import _combined_camera_rows


def config(**values) -> ActivityConfig:
    base = ActivityConfig(
        enabled=True, pose_model="unused.pt", process_every_n_frames=3,
        pose_image_size=416, pose_confidence=0.35, window_seconds=5,
        confirm_ratio=0.70, sleep_min_seconds=25, drink_min_seconds=1.5,
        walking_min_seconds=1.2, standing_min_seconds=1.5,
        desk_min_seconds=2, away_min_seconds=2, movement_threshold=0.035,
        identity_ttl_seconds=15, track_max_distance=140, track_max_missed=1,
        desk_zone_ids=("desk",),
    )
    return replace(base, **values)


def track(track_id=1, box=(100, 100, 220, 360), trail=None) -> TrackSnapshot:
    trail = trail or ((160.0, 360.0), (160.0, 360.0))
    return TrackSnapshot(track_id, box, trail[-1], None, "Chưa xác định", tuple(trail))


def pose(box=(100, 100, 220, 360), *, drinking=False, sleeping=False, standing=False):
    points = {}
    if drinking:
        points = {
            "nose": (160, 135, .95), "left_shoulder": (145, 175, .92),
            "right_shoulder": (175, 175, .92), "left_elbow": (145, 155, .90),
            "left_wrist": (158, 138, .93),
        }
    elif sleeping:
        points = {
            "nose": (160, 181, .93), "left_shoulder": (145, 175, .90),
            "right_shoulder": (175, 175, .90), "left_hip": (145, 250, .90),
            "right_hip": (175, 250, .90),
        }
    elif standing:
        points = {
            "left_shoulder": (145, 150, .90), "right_shoulder": (175, 150, .90),
            "left_hip": (145, 220, .90), "right_hip": (175, 220, .90),
            "left_knee": (145, 290, .90), "right_knee": (175, 290, .90),
            "left_ankle": (145, 350, .90), "right_ankle": (175, 350, .90),
        }
    return PoseObservation(box, .9, points)


def classify(item, item_pose=None, *, in_desk=False, was_at_desk=False, cfg=None):
    return classify_activity(
        item, item_pose, frame_width=800, frame_height=600,
        in_desk=in_desk, was_at_desk=was_at_desk, config=cfg or config(),
    )


def test_case_01_stable_person_in_desk_is_at_desk():
    assert classify(track(), pose(), in_desk=True).activity == ActivityType.AT_DESK


def test_case_02_person_moving_left_to_right_is_walking():
    moving = track(trail=((100, 350), (180, 350), (260, 350)))
    assert classify(moving).activity == ActivityType.WALKING


def test_case_03_stable_vertical_pose_outside_desk_is_standing():
    assert classify(track(), pose(standing=True)).activity == ActivityType.STANDING


def test_case_04_one_drinking_frame_is_not_committed():
    cfg = config(drink_min_seconds=1.5)
    state = TemporalActivityState()
    changed = smooth_activity(state, RuleResult(ActivityType.DRINKING, .8), 10.0, cfg)
    assert not changed and state.current == ActivityType.UNKNOWN


def test_case_05_sustained_drinking_is_committed():
    cfg = config(drink_min_seconds=1.0)
    state = TemporalActivityState()
    for now in (10.0, 10.5, 11.1):
        smooth_activity(state, RuleResult(ActivityType.DRINKING, .8), now, cfg)
    assert state.current == ActivityType.DRINKING


def test_case_06_head_down_for_three_seconds_is_not_sleeping():
    cfg = config(sleep_min_seconds=25)
    state = TemporalActivityState()
    for now in (0.0, 1.0, 2.0, 3.0):
        smooth_activity(state, RuleResult(ActivityType.SLEEPING_SUSPECTED, .75), now, cfg)
    assert state.current != ActivityType.SLEEPING_SUSPECTED


def test_case_07_head_down_and_still_long_enough_is_sleep_suspected():
    result = classify(track(), pose(sleeping=True), in_desk=True)
    assert result.activity == ActivityType.SLEEPING_SUSPECTED
    cfg = config(sleep_min_seconds=25, window_seconds=5)
    state = TemporalActivityState()
    for now in range(0, 27):
        smooth_activity(state, result, float(now), cfg)
    assert state.current == ActivityType.SLEEPING_SUSPECTED


def test_case_08_alternating_frames_do_not_make_ui_jump():
    cfg = config(desk_min_seconds=1, walking_min_seconds=1)
    state = TemporalActivityState()
    for index, now in enumerate((10.0, 10.2, 10.4, 10.6, 10.8)):
        value = ActivityType.AT_DESK if index % 2 == 0 else ActivityType.WALKING
        smooth_activity(state, RuleResult(value, .8), now, cfg)
    assert state.current == ActivityType.UNKNOWN


class FakeRepository:
    def __init__(self):
        self.transitions = []
        self.closed = []
        self.bound = []

    def transition_activity(self, *args): self.transitions.append(args)
    def close_activity(self, camera_id, track_id, at): self.closed.append((camera_id, track_id))
    def bind_activity_identity(self, *args): self.bound.append(args)


class FakeCamera:
    def __init__(self):
        self.frame = np.zeros((600, 800, 3), dtype=np.uint8)
        self.connected = True
        self.capture_fps = 15.0
        self.frame_age_ms = 0
    def read(self): return True, self.frame.copy()
    def start(self): return self
    def stop(self): pass


def runtime(camera_id="cam-a", cfg=None):
    repo = FakeRepository()
    desk = Zone("desk", "Desk", ((0, 0), (1, 0), (1, 1), (0, 1)), (0, 255, 0))
    return ActivityRuntime(camera_id, FakeCamera(), object(), repo, (desk,), cfg or config(
        desk_min_seconds=0, confirm_ratio=.5,
    )), repo


def test_case_09_face_identity_is_attached_to_correct_track():
    engine, _ = runtime()
    engine.submit_identities([IdentityObservation((125, 110, 180, 180), "NV005", "Nguyễn A")])
    engine._process([pose()], 800, 600, 10.0)
    assert engine.latest()[0].employee_id == "NV005"


def test_case_10_identity_cache_survives_temporarily_missing_face():
    engine, _ = runtime()
    engine.submit_identities([IdentityObservation((125, 110, 180, 180), "NV005", "Nguyễn A")])
    engine._process([pose()], 800, 600, 10.0)
    engine._process([pose(box=(102, 100, 222, 360))], 800, 600, 11.0)
    assert engine.latest()[0].employee_id == "NV005"


def test_case_11_lost_track_closes_open_activity_event():
    engine, repo = runtime(cfg=config(desk_min_seconds=0, confirm_ratio=.5, track_max_missed=1))
    engine._process([pose()], 800, 600, 10.0)
    engine._process([], 800, 600, 11.0)
    engine._process([], 800, 600, 12.0)
    assert repo.closed == [("cam-a", 1)]


def test_case_12_activity_overlay_failure_does_not_break_attendance_frame():
    class BrokenActivity:
        def latest(self): return []
        def draw(self, frame): raise RuntimeError("pose failed")
    engine = LiveAttendanceEngine(FakeCamera(), object(), object(), object(), BrokenActivity())
    frame, detections = engine.latest()
    assert frame is not None and detections == []


def test_case_13_and_14_repository_stores_transitions_not_frames(tmp_path):
    repo = SpatialRepository(tmp_path / "spatial.db")
    now = datetime(2026, 9, 23, 9, 0, tzinfo=timezone.utc)
    repo.transition_activity("cam", 7, "NV005", "A", "AT_DESK", .8, "desk", now)
    repo.transition_activity("cam", 7, "NV005", "A", "AT_DESK", .9, "desk", now + timedelta(seconds=1))
    assert len(repo.activity_rows(now - timedelta(seconds=1))) == 1
    repo.transition_activity("cam", 7, "NV005", "A", "WALKING", .75, None, now + timedelta(seconds=10))
    rows = repo.activity_rows(now - timedelta(seconds=1))
    assert len(rows) == 2
    assert next(row for row in rows if row["activity_type"] == "AT_DESK")["duration_seconds"] == 10


def test_case_15_two_people_keep_separate_activity_state():
    engine, _ = runtime()
    poses = [pose((80, 100, 200, 360)), pose((500, 100, 620, 360), standing=True)]
    engine._process(poses, 800, 600, 10.0)
    assert {item.track_id for item in engine.latest()} == {1, 2}


def test_case_16_camera_id_is_part_of_activity_state_and_event_key():
    first, repo_a = runtime("cam-a")
    second, repo_b = runtime("cam-b")
    first._process([pose()], 800, 600, 10.0)
    second._process([pose()], 800, 600, 10.0)
    assert first.latest()[0].camera_id == "cam-a"
    assert second.latest()[0].camera_id == "cam-b"
    assert repo_a.transitions[0][0] == "cam-a" and repo_b.transitions[0][0] == "cam-b"


def test_away_is_spatial_state_not_attendance_state():
    result = classify(track(), None, was_at_desk=True)
    assert result.activity == ActivityType.AWAY


def test_live_table_merges_face_and_activity_without_similarity_column():
    detection = DetectionView((125, 110, 180, 180), "NV005", "Nguyễn A", .82, "CHECK-IN SUCCESS")
    activity = ActivityView(
        "cam-a", 7, (100, 100, 220, 360), "NV005", "Nguyễn A",
        ActivityType.AT_DESK, "Tại bàn", .81, "desk",
        datetime.now(timezone.utc), 12.0,
    )
    rows = _combined_camera_rows([detection], [activity])
    assert len(rows) == 1
    assert set(rows[0]) == {"employee", "result", "activity", "duration"}
    assert rows[0]["activity"].label == "Tại bàn"
