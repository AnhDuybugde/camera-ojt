# Implementation status

## Delivered in this refactor

- One `camera-ojt` launcher owns the backend API, camera runtime and Streamlit
  attendance UI. Imou and webcam remain profiles of the shared runtime; old
  scripts are compatibility entrypoints.
- The UI calls an authenticated allowlisted API. Server-side sessions derive
  role and employee scope; pipeline frames use signed short-lived URLs.
- Attendance events have stable IDs and are durably queued for ingestion.
  Ambiguous/unknown passages are retained for review rather than assigned to a
  guessed employee. Face results from stale track generations are rejected.
- Employee data, versioned enrollment samples, attendance events and their
  replication outbox now share the attendance SQLite database. Registration
  updates samples, representative vector, audit entry and replication triggers
  in one transaction.
- Supabase schema/migration and revision-aware replication are in place. The
  local-first UI remains usable offline; conflicting cloud edits are surfaced
  for review rather than silently overwriting local data.
- Gemini query processing and Tavily web search use a shared voice-query module.
- React dashboard artifacts were removed from the active layout; Streamlit is
  the supported user interface. Runtime data, credentials, enrollment photos,
  databases and generated invitation/bootstrap files are excluded from Git.

## Data migration state

The 319 versioned enrollment samples previously held in `var/enrollment.db`
were copied idempotently into the main attendance database. The original file
was intentionally retained unchanged as a rollback source. Supabase migration
was applied earlier in this work. A follow-up cloud sync was attempted after
consolidation but network access was unavailable; an escalated upload was
rejected by the safety reviewer because it would export biometric enrollment
data. The local replication outbox remains the safe pending hand-off. Do not
delete either local database until an authorized sync and backup are confirmed.

## Still required before production

- Obtain explicit authorization for sending the 319 biometric vectors to the
  configured Supabase project, then run `python tools/sync_cloud.py` and resolve
  any reported conflicts. Verify Supabase is the intended trusted destination.
- Provision and verify Supabase `camera_memberships` only for users who need
  direct scoped Supabase access; application login currently remains local.
- Run acceptance on labeled holdout footage: identity false assignments, ID
  switches, unknown rate, eligible passage recall, recovery latency, frame age,
  and memory over the two-camera target workload. No accuracy/performance claim
  is made until that measurement is complete.
- Confirm camera-specific zone calibration, direction, schedule rules and
  employee registry mappings in the installation environment.
- Configure and validate external credentials (Gemini, Tavily, Google Sheets,
  Supabase) without placing secrets in source control.

See [architecture](UNIFIED_PLATFORM.md) for service/data contracts and
[roadmap](ROADMAP.md) for gated camera capabilities.
