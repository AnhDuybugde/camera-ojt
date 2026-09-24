"""Read-only camera/AI benchmark; it never calls attendance or writes video."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera.camera_manager import CameraManager
from config import settings
from face.detector import FaceDetector
from spatial.activity_config import activity_settings
from spatial.pose_service import PoseService


def _wait_new(camera: CameraManager, previous: int, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sequence = camera.frame_sequence
        if sequence != previous:
            ok, frame = camera.read()
            if ok and frame is not None:
                return sequence, frame
        time.sleep(0.01)
    return previous, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Index in configured camera list")
    parser.add_argument("--baseline-seconds", type=float, default=4.0)
    parser.add_argument("--pose-frames", type=int, default=8)
    parser.add_argument("--face-frames", type=int, default=3)
    args = parser.parse_args()
    name, source = settings.camera_sources[args.camera]
    camera = CameraManager(source).start()
    try:
        if not camera.wait_for_frame(10):
            print(json.dumps({"camera": name, "error": camera.error or "No frame"}, ensure_ascii=False))
            return 2
        time.sleep(max(1.0, args.baseline_seconds))
        baseline_fps = camera.capture_fps
        sequence, frame = camera.frame_sequence, camera.read()[1]

        pose = PoseService(activity_settings)
        pose_times = []
        pose_people = 0
        for _ in range(max(1, args.pose_frames)):
            sequence, latest = _wait_new(camera, sequence)
            frame = latest if latest is not None else frame
            started = time.perf_counter()
            poses = pose.infer(frame)
            pose_times.append(time.perf_counter() - started)
            pose_people = max(pose_people, len(poses))

        face = FaceDetector()
        face_times = []
        face_count = 0
        for _ in range(max(1, args.face_frames)):
            sequence, latest = _wait_new(camera, sequence)
            frame = latest if latest is not None else frame
            started = time.perf_counter()
            faces = face.detect(frame)
            face_times.append(time.perf_counter() - started)
            face_count = max(face_count, len(faces))

        pose_steady = pose_times[1:] or pose_times  # first call includes model load
        result = {
            "camera": name,
            "source": camera.safe_source,
            "camera_fps_before_ai": round(baseline_fps, 2),
            "camera_fps_during_ai": round(camera.capture_fps, 2),
            "pose_inference_fps": round(1 / (sum(pose_steady) / len(pose_steady)), 2),
            "pose_mean_ms": round(sum(pose_steady) / len(pose_steady) * 1000, 1),
            "pose_first_call_ms": round(pose_times[0] * 1000, 1),
            "face_mean_ms": round(sum(face_times) / len(face_times) * 1000, 1),
            "people_seen": pose_people,
            "faces_seen": face_count,
            "pose_backend": pose.backend_name,
            "note": "Read-only benchmark; attendance was not called.",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
