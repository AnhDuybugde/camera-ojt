# Frontend — Streamlit UI `:8501` (no Node/React in this repo)

The requested `frontend/src/pages|components|services|hooks + package.json`
layout describes a React app. This repo's frontend is **Streamlit**
(`apps/attendance/app.py`), so there is no `package.json`. Mapping:

| Requested (`frontend/src/...`) | Actual code |
|---|---|
| `pages/` | `apps/attendance/ui/` (`dashboard`, `employees`, `history`, `live_attendance`, `login`, `register_face`, `statistics`, `system_integration`, ...) |
| `components/` | `apps/attendance/ui/components.py` + `apps/attendance/ui/theme.py` |
| `services/` | `apps/attendance/integration/backend.py` (`RemoteService` → backend `:8767`) + `apps/attendance/integration/camera_ojt.py` (`CameraOjtClient` → pipeline `:8765`) |
| `hooks/` | `apps/attendance/ui/common.py` (`get_db`, `get_detector`, `get_auth_service`, ... — Streamlit `cache_resource` singletons) |
| `package.json` | n/a — Python deps in `pyproject.toml` (`ui` extra) + `apps/attendance/requirements.txt` |

Key rule: the UI **never touches the database directly** — every page goes
through `RemoteService` to backend `:8767` (`CAMERA_API_URL`).
If a page shows *"Backend chưa chạy. Khởi động bằng camera-ojt run."*,
the backend process is down or `CAMERA_API_URL` points elsewhere.
