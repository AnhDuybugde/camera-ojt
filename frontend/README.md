# Frontend — Streamlit UI `:8501` (no Node/React in this repo)

Canonical code lives here (`frontend/src/...`). Old locations under
`apps/attendance/ui/` + `apps/attendance/integration/` remain as one-line
compatibility shims. Entry point stays `apps/attendance/app.py` (Streamlit
static serving requires `apps/attendance/static/` next to it).

| `frontend/src/...` | Origin (git history preserved) |
|---|---|
| `pages/` (14 pages: dashboard, employees, history, live_attendance, login, register_face, statistics, system_integration, …) | `apps/attendance/ui/*.py` |
| `components/components.py`, `components/theme.py` | `apps/attendance/ui/components.py`, `theme.py` |
| `services/backend.py` (`RemoteService` → backend `:8767`) | `apps/attendance/integration/backend.py` |
| `services/camera_ojt.py` (`CameraOjtClient` → pipeline `:8765`) | `apps/attendance/integration/camera_ojt.py` |
| `hooks/common.py` (`get_db`, `get_detector`, … singletons) | `apps/attendance/ui/common.py` |
| `assets/style.css` | `apps/attendance/assets/style.css` |
| `package.json` | scripts only (`dev`, `smoke`, `check-backend`) — no Node deps |

Key rule: the UI **never touches the database directly** — every page goes
through `RemoteService` to backend `:8767` (`CAMERA_API_URL`).
If a page shows *"Backend chưa chạy. Khởi động bằng camera-ojt run."*,
the backend process is down or `CAMERA_API_URL` points elsewhere.
