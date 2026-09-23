# Unified platform contracts

## Ownership and lifecycle

`camera-ojt run` owns three processes: application API, camera runtime and
Streamlit. The API owns local accounts, attendance, registration, ingestion and
cloud sync. Closing a browser has no effect on ingestion. One workstate module
implements Imou and webcam profiles; old scripts are compatibility entrypoints.

The legacy business modules remain in `apps/attendance` during migration; only
the backend instantiates their persistent services. UI clients expose an
allowlisted API rather than arbitrary SQL or Python reflection.

## API

`POST /api/v1/{service}/{operation}` accepts `{args: [], kwargs: {}}` and returns
`{result: ...}`. Services: `auth`, `employees`, `attendance`, `enrollment`,
`operations`, `sync`. Authentication uses backend-issued bearer sessions,
1-hour expiry. The API binds loopback by default; use a TLS reverse proxy
if deploying beyond this machine.

Pipeline HTTP is separately protected by an internal bearer token. Only signed
short-lived image URLs are given to browser clients. Legacy loopback-only
pipeline launches can run without that token; network exposure requires it.

## Identity

Track IDs are temporary; employee IDs are permanent. Async jobs carry the
identity generation; retired, reassigned, old and occluded results are discarded.
Suspected overlap revokes the employee binding and freezes appearance updates.
A new confirmed face is needed before publishing a name again.

Source enrollments are stored per image and model space in SQLite. Matching is
vectorized in RAM. All accepted samples remain persisted; unversioned `.npy`
runtime prototypes are not silently trusted. Replacing enrollment is atomic
within its sample store. The legacy representative-vector projection in the
employee database is retained for compatibility; the sample store is authoritative.

## Events and cloud

Door observations have UUID event IDs, UTC observation timestamps, a runtime
session, camera, direction and evidence. Daily attendance uses Asia/Ho_Chi_Minh.
Events and their outbox entries commit with the daily projection. Retries use
the same event ID. Unknown passages are retained for review; review creates a
new deterministic correction event referencing the original.

SQLite triggers queue changes to employees, schedules, daily attendance, audit
and enrollment. Supabase compare-and-swap advances revisions; differing newer
cloud records become explicit conflicts. Identical retries are idempotent. A
transaction lock orders cloud sequence numbers by commit so incremental readers
do not miss concurrent commits. Batch size is 100; provider calls are outside
local SQLite transactions. No passwords or authentication tables are replicated.

New cloud tables coexist with the old Supabase schema. `camera_memberships`
links Supabase Auth users to employee IDs and roles. App login uses Supabase
email/password or email OTP; the service key and Auth access token stay on the
backend, which issues its own short-lived role-scoped session. Create explicit
membership mappings from an approved roster CSV with
`tools/provision_supabase_users.py`; the template is
`tools/templates/supabase-users.csv.example`. No local fallback account is
accepted when Supabase Auth is configured.

`tools/sync_cloud.py` synchronizes business tables and events, but biometric
samples are excluded by default. They require both an explicit command-line
flag and `CAMERA_SYNC_BIOMETRIC=1` on the backend. Enable that only after the
project storage region, applicable consent/retention rules, and transfer
requirements have been reviewed. Never commit `.env` or real roster CSVs.

The UI state stream is a latest-value snapshot (2 Hz); intermediate state
changes are coalesced. Durable attendance/event writes use the local outbox:
stable daily state writes include a 30-second heartbeat, while distinct events
remain FIFO and are not coalesced. Unknown people are shown as an aggregate
count and do not create employee attendance records.

## Recovery

Before migration run `tools/migrate_local.py`; backups use SQLite backup API
and include committed WAL data. To roll back, stop all processes and restore
the chosen databases and prior code together. Keep cloud tables until rollback
is validated; the migration does not require deleting legacy data.

Invalid/unmapped legacy queue rows are retained in `rejected_writes`; resolved
enrollment mappings can restore them. Do not delete queue or enrollment databases
to fix sync errors. Resolve conflicts through the UI and inspect diagnostics.
