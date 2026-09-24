# Production V2 changes

- Fixed monitor-only bug: disabling attendance no longer disables face recognition.
- Production automatic attendance now fails closed without a configured liveness gate.
- Added HTTP liveness provider adapter with score threshold + timeout.
- Added deterministic AttendanceLifecycleEngine and tests.
- Added attendance provenance fields and audit-log schema.
- Attendance write is constrained to the configured entry camera (`ATTENDANCE_ENTRY_CHANNEL`, default B).
- Added Gemini Live assistant (`be_xinh_live_assistant.py`) using native audio and local tool calling.
- Added incremental 24 kHz PCM -> IMOU VisualTalk streaming path; response can start before the full answer exists.
- Live assistant keeps one conversation session after wake, allowing natural follow-up questions without repeating the wake phrase.
- Classic assistant now includes short conversation history as fallback context.
- Dashboard and PowerShell launchers read the selected mode from `backend/.env`; Classic remains the safe default and Live is an explicit opt-in.
- Updated production preflight for liveness and realtime voice.
- Corrected explicit CUDA validation order so deployment errors identify missing `onnxruntime-gpu` correctly.
- Liveness provider I/O runs in the background FaceWorker so it cannot stall the realtime camera loop.
- Attendance lifecycle is restored from the database after restart and ignores repeated room-state events.
- Backend regression: 332/332 tests pass in review environment.
