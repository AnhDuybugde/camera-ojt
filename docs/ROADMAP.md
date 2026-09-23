# Camera capability backlog

Ship only after identity/attendance acceptance on labeled holdout video:

1. Occupancy and dwell time by configurable zone; aggregate anonymous counts.
2. Camera health: disconnected, covered lens, changed viewpoint, stale frame.
3. After-hours presence and restricted-zone alerts, with deduplication and evidence.
4. Vietnamese event search by time/camera/zone, enforcing the same account permissions.
5. Separate experiments for abandoned objects and falls; collect and evaluate
   representative data before enabling alerts.

Performance acceptance workload: two camera streams, up to five concurrent
people and thirty enrolled employees. Report false assignments, ID switches,
unknown rate, eligible-passage recall, recovery latency, p95 frame age and memory
growth over eight hours. Keep calibration clips separate from holdout clips;
never meet a false-assignment target by returning Unknown for everyone.
