# Production deployment baseline

This document captures the minimum hardening required before exposing the camera system outside a developer machine.

## 1. Network boundary

The camera MJPEG/status server and pipeline supervisor must bind to `127.0.0.1`.
Remote users should reach them only through a reverse proxy/VPN. The runtime now refuses non-loopback binding unless an explicit escape hatch is set:

- `CAMERA_ALLOW_REMOTE_STREAM=1`
- `CAMERA_ALLOW_REMOTE_CONTROL=1`

Do not set these in production unless the host is protected by another access-control layer.

## 2. CORS

Set `CAMERA_CORS_ORIGINS` to the exact dashboard origins, comma-separated. Example:

```env
CAMERA_CORS_ORIGINS=https://camera.example.internal
```

Wildcard CORS is intentionally not the default.

## 3. Database migration

Run:

```text
backend/supabase/migrations/20260924_production_hardening.sql
```

This adds `attendance_daily.check_out_at`, `room_status_daily.merged_into`, and lifecycle event values required for the next attendance phase.

## 4. Attendance write semantics

`/attendance/send` is state-changing and is therefore POST-only. The dashboard has been updated accordingly.

## 5. Counting semantics

`count_a` and `count_b` are camera-local visible-box counts. `count` is the
fresh canonical in-room count and must be used by the conversational agent when
answering “how many people are in the room?”. The API also exposes:

- `count_visible`: unique identities visible in the current camera frames.
- `count_logical`: canonical identities still marked in-room before freshness filtering.
- `count_stale`: logical tracks older than `OCCUPANCY_STALE_SECONDS`.

This prevents an old track from inflating the live occupancy while preserving
enough information for audit and tuning. The office-camera baseline is 15
seconds; tune it from recorded occlusion/exit tests rather than increasing it to
hide identity churn.

## 6. Authentication and sessions

Set `APP_ENV=production` in both `backend/.env` and `frontend/.env`. New auth
databases require strong values for `ADMIN_BOOTSTRAP_PASSWORD` and
`EMPLOYEE_BOOTSTRAP_PASSWORD`. Existing accounts must be rotated through the
admin interface; the production preflight detects the known legacy passwords
without printing any hash or secret.

Password resets now create a one-time random temporary password rather than
resetting every employee to the same value. Authenticated UI sessions expire
after `SESSION_IDLE_TIMEOUT_MINUTES` (30 by default).

## 7. Health and observability

- `GET /healthz`: process liveness; does not include employee or image data.
- `GET /readyz`: returns HTTP 200 only when status and at least one camera frame
  are fresh; otherwise HTTP 503.
- `GET /status.json`: includes `inference_ms`, `loop_fps`, `runtime_metrics` and
  `status_generated_at` for diagnosis.

Use `/readyz` for a reverse-proxy health check. Do not use `/status.json` for a
public health probe because it contains operational/person data.

## 8. Safe biometric rollout

Production defaults to `ATTENDANCE_AUTOMATION_ENABLED=false`. In this mode face
recognition and the observation dashboard continue running, but the backend does
not create automatic attendance records. This is the safe deployment mode until
a real, measured liveness/anti-spoof strategy is wired into the backend gate.

Declaring a provider name in an environment variable is not an integration. Do
not enable automatic biometric writes until spoof tests, false-accept/false-reject
measurements, consent, retention and manual correction workflows have passed.

## 9. Release gate

Before every release run:

```powershell
cd D:\team-integration\backend
.\.venv\Scripts\python.exe scripts\preflight_production.py --strict
```

Exit code `0` is the release gate. CI also runs backend tests, frontend tests and
the React dashboard build on every push and pull request.

## 10. Remaining release blockers

Before biometric attendance is considered production-ready:

1. Integrate a real liveness/anti-spoof provider before enabling automatic attendance.
2. Move CHECK-IN/CHECK-OUT/RETURN/TEMP_OUT into one backend attendance lifecycle service.
3. Calibrate face thresholds on deployment-camera data; do not copy thresholds between sites.
4. Put the local services behind authenticated TLS reverse proxy/VPN.
5. Add retention/deletion policy for face crops, embeddings, video clips and audit logs.
6. Run a staged pilot with false-accept / false-reject measurement before enabling automatic actions such as door unlock.
