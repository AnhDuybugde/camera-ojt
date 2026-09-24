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

`count_a` and `count_b` are camera-local visible-box counts. `count` is the canonical in-room count derived from global IDs and must be used by the conversational agent when answering “how many people are in the room?”.

## 6. Remaining release blockers

Before biometric attendance is considered production-ready:

1. Integrate a real liveness/anti-spoof provider and fail closed when attendance requires it.
2. Move CHECK-IN/CHECK-OUT/RETURN/TEMP_OUT into one backend attendance lifecycle service.
3. Calibrate face thresholds on deployment-camera data; do not copy thresholds between sites.
4. Put the local services behind authenticated TLS reverse proxy/VPN.
5. Add retention/deletion policy for face crops, embeddings, video clips and audit logs.
6. Run a staged pilot with false-accept / false-reject measurement before enabling automatic actions such as door unlock.
