# Identity architecture

Production has three identifiers with different lifetimes:

- `local_track_id`: ByteTrack ID scoped to one camera process.
- `global_id`: promoted person entity scoped to the local calendar day.
- `employee_id`: durable employee identity, assigned only by confirmed face consensus.

The runtime batches camera A and B at one timestamp. A global identity stores a
separate sighting per camera, so overlap is valid and does not create two
employees. New local tracks remain tentative for five observations and do not
consume a public GID.

OSNet is required for production continuity. InsightFace is used only to bind an
employee: score `>= 0.70`, top-1/top-2 margin `>= 0.10`, and three consecutive
matching observations inside three seconds. A committed employee binding cannot
be stolen by another GID. Uncertain observations remain `UNKNOWN`.

## Replay acceptance

Run the pipeline on synchronized files and save its trace:

```powershell
python scripts/run_workstate.py --source-a cam-a.mp4 --source-b cam-b.mp4 `
  --identity-log output/predictions.csv
```

Truth CSV columns are:

```text
frame,camera,truth_id,employee_id,x1,y1,x2,y2
```

Evaluate the trace:

```powershell
python scripts/evaluate_replay.py `
  --truth annotations.csv --predictions output/predictions.csv
```

Exit code `2` means a false employee assignment or uniqueness violation. The
holdout acceptance target is zero for both. Keep calibration and holdout video
segments separate.

Generate conservative threshold suggestions from a labeled score CSV:

```powershell
python scripts/calibrate_identity_thresholds.py --scores calibration-scores.csv
```

Identity state is stored locally in `output/identity_state.db`. Restarting on the
same day restores confirmed identities; the namespace resets at local midnight.
