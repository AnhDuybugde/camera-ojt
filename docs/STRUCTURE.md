# Repo structure — `camera-attendance/` layout (real code, not facades)

The repo root **is** the `camera-attendance/` project (named `camera-ojt`
from history). Canonical code lives in `backend/` + `frontend/`; old paths
remain as one-line compatibility shims (re-exporting the new modules) so
existing imports, tests and `camera-ojt run` keep working.

```text
camera-attendance/
├── frontend/                  # Streamlit UI (canonical)
│   ├── src/
│   │   ├── pages/             # 14 pages (dashboard, live_attendance, login, …)
│   │   ├── components/        # components.py, theme.py
│   │   ├── services/          # backend.py (RemoteService :8767), camera_ojt.py (:8765)
│   │   ├── hooks/             # common.py (cache_resource singletons)
│   │   └── assets/            # style.css
│   ├── package.json           # scripts only (dev, smoke, check-backend)
│   └── README.md
├── backend/                   # business services + API (canonical)
│   ├── app/
│   │   ├── api/               # server.py (:8767), service.py, codec.py, access.py,
│   │   │                      # operations.py, enrollment.py, supabase_auth.py
│   │   │                      # + attendance.py, members.py, rooms.py (domain routers)
│   │   ├── camera/            # stream.py (MJPEG :8765), capture.py
│   │   ├── recognition/       # face_detector.py, face_recognition.py, tracker.py
│   │   ├── services/          # attendance_service.py, attendance_admin_service.py,
│   │   │                      # attendance_rules.py, room_status_service.py
│   │   ├── database/          # database.py, models.py
│   │   └── main.py            # load_services + run_backend
│   ├── requirements.txt       # backend install (pip install -r backend/requirements.txt)
│   └── README.md
├── apps/attendance/           # Streamlit entry (app.py) + remaining cores
│                              # (config, auth, google_sheets, spatial, utils, camera,
│                              #  face/embedding, static/) + compat shims
├── src/camera_tracking/       # pipeline runtime (vision/tracking/voice/store/workstate)
│                              # + compat shims for moved modules
├── models/                    # face_recognition packs, ReID cache, STT
├── data/                      # faces|images|samples (+ logs in apps/attendance/logs/)
├── config/
│   └── cameras.yaml           # camera inventory (A=room, B=door) + port map
├── docker-compose.yml         # backend :8767 + ui :8501 (+ pipeline :8765 profile)
├── Dockerfile                 # backend/ui image
├── .gitignore
└── README.md
```

Install notes: `backend` + `frontend` are real Python packages installed
editable (`pip install -e .` from root), so `import backend…` / `import
frontend…` works from any cwd — no `sys.path` hacks in entry points.

Connection flow (see `tools/check_backend_connection.py`):

```text
Streamlit UI :8501 ──RemoteService──▶ backend :8767 ──▶ SQLite (+ Supabase outbox)
        │                                ▲
        │ CameraOjtClient                │ drain queue (WriteQueue output/queue.db)
        ▼                                │
Pipeline :8765 (/status.json, MJPEG) ────┘
```

Ports: backend `8767`, stream `8765`, supervisor `8766`, UI `8501`.
One shared `CAMERA_INTERNAL_TOKEN` per `camera-ojt run` authenticates
pipeline access; `CAMERA_API_URL` points the UI at the backend.
