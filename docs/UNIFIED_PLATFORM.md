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
8-hour expiry and account-version revocation. Role and employee ownership are
derived server-side; caller-supplied roles do not grant access. Password hashes
are never returned. The API binds loopback by default; use a TLS reverse proxy
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
links Supabase users to employee IDs for scoped direct reads. Application users
authenticate locally, so the app also works without Internet. Provision cloud
memberships explicitly before offering direct Supabase user access.

## Recovery

Before migration run `tools/migrate_local.py`; backups use SQLite backup API
and include committed WAL data. To roll back, stop all processes and restore
the chosen databases and prior code together. Keep cloud tables until rollback
is validated; the migration does not require deleting legacy data.

Invalid/unmapped legacy queue rows are retained in `rejected_writes`; resolved
enrollment mappings can restore them. Do not delete queue or enrollment databases
to fix sync errors. Resolve conflicts through the UI and inspect diagnostics.
