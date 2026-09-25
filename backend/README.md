# Backend — authoritative business services + JSON API `:8767`

Canonical code lives here (`backend/app/...`). Old locations remain as
one-line compatibility shims re-exporting these modules.

| `backend/app/...` | Origin (git history preserved) |
|---|---|
| `api/server.py`, `service.py`, `codec.py`, `access.py`, `operations.py`, `enrollment.py`, `supabase_auth.py` | `src/camera_tracking/api/` |
| `api/attendance.py`, `api/members.py`, `api/rooms.py` | new domain routers over the API (typed `call(service, method, …)` clients) |
| `api/_http.py` | shared stdlib HTTP transport for the routers |
| `camera/stream.py` | `src/camera_tracking/streaming/mjpeg.py` (pipeline `:8765`) |
| `camera/capture.py` | `src/camera_tracking/camera/source.py` (`camera_imou.py` stays for RTSP) |
| `recognition/face_detector.py`, `face_recognition.py` | `apps/attendance/face/detector.py`, `recognizer.py` |
| `recognition/tracker.py` | `src/camera_tracking/tracking/byte.py` (ByteTrack) |
| `services/attendance_service.py`, `attendance_admin_service.py`, `attendance_rules.py` | `apps/attendance/attendance/` |
| `services/room_status_service.py` | `src/camera_tracking/workstate/room_fusion.py` (room status fusion) |
| `database/database.py`, `models.py` | `apps/attendance/database/db.py`, `models.py` |
| `main.py` | `src/camera_tracking/application/backend.py` (`load_services` + `run_backend`) |

Run it:

```bash
camera-ojt backend          # business API :8767 only
camera-ojt run --no-camera  # backend :8767 + UI :8501, no camera
camera-ojt run              # full stack: backend + pipeline :8765 + UI
camera-ojt doctor           # modules + keys + port status + backend reachability
python tools/check_backend_connection.py  # end-to-end healthcheck
```

Install: `pip install -r backend/requirements.txt` (from repo root, after torch).
Config: `.env` (secrets) + `config/default.yaml` (pipeline) +
`config/cameras.yaml` (camera inventory). Data: `data/faces|images`,
`models/`, `output/`, `var/` (all git-ignored except `.gitkeep`).
