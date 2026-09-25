# Repo structure — requested `camera-attendance/` layout vs actual paths

The repo root **is** the `camera-attendance/` project (named `camera-ojt`
from history). Code was deliberately **not moved** so imports, tests and
`camera-ojt run` keep working. Use this table to navigate:

```text
camera-attendance/
├── frontend/                  # facade docs → real UI in apps/attendance/
│   ├── src/
│   │   ├── pages/             # → apps/attendance/ui/*.py
│   │   ├── components/        # → apps/attendance/ui/components.py, theme.py
│   │   ├── services/          # → apps/attendance/integration/backend.py (RemoteService :8767)
│   │   └── hooks/             # → apps/attendance/ui/common.py (cache_resource singletons)
│   └── README.md              # (no package.json — Streamlit, not React)
├── backend/                   # facade docs → real backend code
│   ├── app/
│   │   ├── api/               # → src/camera_tracking/api/ (server.py :8767, service.py)
│   │   ├── camera/            # → src/camera_tracking/camera/ + streaming/mjpeg.py (:8765)
│   │   ├── recognition/       # → src/camera_tracking/face/ + tracking/ + apps/attendance/face/
│   │   ├── services/          # → apps/attendance/attendance/ + src/camera_tracking/workstate/
│   │   ├── database/          # → apps/attendance/database/
│   │   └── main.py            # → src/camera_tracking/application/backend.py (run_backend)
│   └── README.md
├── models/                    # → models/ (face_recognition packs, ReID cache, STT)
├── data/                      # → data/faces|images|samples (+ logs in apps/attendance/logs/)
├── config/
│   └── cameras.yaml           # camera inventory (A=room, B=door) + port map
├── docker-compose.yml         # backend :8767 + ui :8501 (+ pipeline :8765 profile)
├── Dockerfile                 # backend/ui image
├── .gitignore
└── README.md
```

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
