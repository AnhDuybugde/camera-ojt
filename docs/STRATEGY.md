# Project Strategy: Stable Global Person ID Across Two Cameras

## 1. Goal

Keep **one stable Global Person ID** for each person across:

- many short tracklets (detector flicker, occlusion),
- long disappearances from the camera,
- **both channels** (two IMOU cameras, channel 1 and channel 2).

The system answers **"who is this person?"** with a Global ID instead of
relying on ByteTrack's temporary per-channel `track_id`.

## 2. Pipeline (per channel, running independently)

```text
Channel A ─┐
           ├─ YOLO26s (person only) → ByteTrack → GlobalIdentityManager ─┐
Channel B ─┘  (short-term tracking)    (shared, cross-channel IDs)        ├─► overlay (box + G<ID> + business state)
                                                                         └─► console summary
```

| Stage | Role | Code |
|---|---|---|
| Detection | YOLO26s, `person` class only, auto device (`cuda`/`mps`/`cpu`) | `detection/yolo.py` |
| Short-term tracking | ByteTrack per channel: motion prediction, IoU matching, data association, track buffer on brief detection gaps | `tracking/byte.py` |
| **Global identity** | One shared manager maps tracklets → stable Global IDs across channels and long gaps | `tracking/global_identity.py` |
| Business (display only) | Per-channel status: `WORKING` / `AWAY_TEMP` / `POSSIBLY_OUT` / `RETURNING` | `workstate/channel_status.py` |
| Visualization | Boxes + `G<ID>` labels + business state + person count | `visualization/people.py` |

## 3. How the Global Identity Manager works

1. **Appearance gallery** — each identity stores several good Re-ID embeddings
   (histogram of clothing colors; only confident, well-sized crops are kept),
   not a single feature vector. Matching uses the best gallery similarity,
   which is robust to pose changes.
2. **Combined score** — appearance similarity + observation history + time
   since last seen + appear/disappear positions + current channel +
   spatio-temporal constraints.
3. **Soft channel evidence only** — movement priors between channel 1 and 2
   add a tiny bonus/penalty (weight `0.10`). They are never a mandatory route.
4. **Global association** — all current tracks vs. all known identities form a
   **cost matrix**, solved with the **Hungarian algorithm** (not per-detection
   argmax), after **gating** removes implausible pairs (different appearance,
   impossible same-camera jump).
5. **Identity lifecycle, never abrupt deletion** —
   `ACTIVE → TEMP_LOST → LONG_LOST → UNRESOLVED`.
   A reappearing person reconnects to the old Global ID, even on the other
   channel. Only `UNRESOLVED` identities past the keep window (`300 s`
   default) are purged.

## 4. Separation of concerns (important)

- **Tracking / identity** answers *"which person is this?"* → Global ID.
- **Channel business logic** answers *"what is that person doing here?"*
  (working, temporarily away, possibly out, returning).

The business layer only **reads** Global IDs + presence. It can never write
back into the identity manager, so a business-rule mistake cannot corrupt IDs.
Channel-specific rules (e.g. seat ROIs for channel 1) can be plugged in later
without touching identity code.

## 5. Configuration

Identity behaviour is tuned in `config/default.yaml`, block `identity:`:

```yaml
identity:
  gallery_size: 8
  match_threshold: 0.40
  temp_lost_s: 5.0
  long_lost_s: 60.0
  unresolved_keep_s: 300.0
```

Camera streams come from `.env` (`IMOU_IP` / `IMOU_USER` / `IMOU_PASSWORD`).

## 6. Test with a single command

Headless smoke test on both live cameras (30 processed frames, no window needed):

```powershell
python scripts\run_workstate.py --max-frames 30
```

Expected output: both RTSP streams open, and a stable `Global IDs: [...]`
summary at the end, e.g. `G1: state=... last_channel=A hits=... gallery=8`.
For the live view with windows, add `--display` (press `q` to quit).
Unit tests: `python -m pytest -q`.
