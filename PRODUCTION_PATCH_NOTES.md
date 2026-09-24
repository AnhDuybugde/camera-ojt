# Production hardening v1 — 2026-09-24

## Implemented

- Camera stream defaults to `127.0.0.1` in `run.ps1`.
- Streamlit UI also defaults to `127.0.0.1` via `-UiHost`.
- MJPEG service refuses non-loopback binding unless `CAMERA_ALLOW_REMOTE_STREAM=1` is explicit.
- Pipeline supervisor refuses non-loopback binding unless `CAMERA_ALLOW_REMOTE_CONTROL=1` is explicit.
- Wildcard CORS removed from camera/supervisor JSON APIs; use `CAMERA_CORS_ORIGINS`.
- `/attendance/send` changed from state-changing GET to POST; React dashboard updated.
- Added no-store/nosniff headers to JSON APIs.
- Fixed canonical room count: `count` now derives from canonical Global IDs; `count_a`/`count_b` remain camera-local counts.
- Streamlit “Tổng hiện diện” now prefers canonical `count`.
- Supabase schema now includes `attendance_daily.check_out_at` and `room_status_daily.merged_into`.
- Added repeatable migration `backend/supabase/migrations/20260924_production_hardening.sql`.
- Added lifecycle event values `CHECK_OUT` and `TEMP_OUT` to schema for the next attendance phase.
- Added `backend/scripts/preflight_production.py` release preflight.
- Added production network/env documentation.
- Added tests covering remote-stream fail-safe and POST-only attendance send.

## Verification

Backend regression with `PYTHONPATH=src:. pytest -q`:

- 317 passed
- 1 pre-existing environment-dependent failure (`insightface` is not installed in the review environment, so the CUDA provider test exits before reaching its intended assertion)

New focused tests:

- 5 passed (`test_stream_security.py` + `test_supervisor.py`)

Frontend full pytest could not be executed in the review container because its optional runtime packages (`streamlit`, `bcrypt`) are not installed. The modified Streamlit module passes Python compilation.

## Still release blockers

1. Wire a real liveness/anti-spoof provider into the backend attendance decision and fail closed when required.
2. Implement one backend attendance lifecycle service for CHECK_IN → TEMP_OUT/RETURN → CHECK_OUT.
3. Run the Supabase migration and make Supabase/PostgreSQL the authoritative attendance source.
4. Calibrate face thresholds and temporal consensus using deployment-camera data.
5. Put localhost services behind authenticated TLS reverse proxy/VPN for remote access.
6. Define retention/deletion rules for biometric crops, embeddings, clips and audit logs.
7. Pilot with FAR/FRR, ID-switch rate, checkout correctness and end-to-end latency metrics before automatic physical access control.
