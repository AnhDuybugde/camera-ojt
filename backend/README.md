# Backend — authoritative business services + JSON API `:8767`

This folder is the **clear entry point** for everything the requested
`backend/app/...` layout covers. The implementation lives in the existing
Python packages (no code was moved, so imports/tests keep working):

| Requested (`backend/app/...`) | Actual code |
|---|---|
| `api/attendance.py`, `api/members.py`, `api/rooms.py` | `src/camera_tracking/api/service.py` (`ApplicationAPI`: `employees`/`attendance`/`sync`/`operations`/`enrollment` services) + `src/camera_tracking/api/server.py` (HTTP `:8767`) |
| `camera/stream.py`, `camera/capture.py` | `src/camera_tracking/camera/` + `src/camera_tracking/streaming/mjpeg.py` (pipeline `:8765`) + `src/camera_tracking/application/workstate.py` (runtime) |
| `recognition/face_detector.py`, `face_recognition.py`, `tracker.py` | `apps/attendance/face/detector.py` + `src/camera_tracking/face/` + `src/camera_tracking/tracking/` |
| `services/attendance_service.py`, `room_status_service.py` | `apps/attendance/attendance/` + `src/camera_tracking/workstate/` (door/room fusion) |
| `database/models.py`, `database.py` | `apps/attendance/database/` |
| `main.py` | `src/camera_tracking/application/backend.py` (`run_backend`) via `camera-ojt backend` / `camera-ojt run` |

Run it:

```bash
camera-ojt backend          # business API :8767 only
camera-ojt run --no-camera  # backend :8767 + UI :8501, no camera
camera-ojt run              # full stack: backend + pipeline :8765 + UI
camera-ojt doctor           # modules + keys + port status + backend reachability
python tools/check_backend_connection.py  # end-to-end healthcheck
```

Config: `.env` (secrets) + `config/default.yaml` (pipeline) +
`config/cameras.yaml` (camera inventory). Data: `data/faces|images`,
`models/`, `output/`, `var/` (all git-ignored except `.gitkeep`).
