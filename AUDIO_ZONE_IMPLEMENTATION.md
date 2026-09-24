# Bé Xinh Audio Zone Implementation

## Scope

This revision changes only the audio/integration boundary and its tests/docs.
It does **not** modify the web UI, YOLO, ByteTrack, Re-ID, face recognition,
zone geometry, or camera-opening logic.

## Implemented behavior

- `DOOR_ENTER`: personalised greeting when identity is known, generic otherwise.
- `DOOR_EXIT`: playful leaving line without claiming the person is going home.
- `BE_XINH_NEAR`: playful interaction near Bé Xinh.
- `WATER`: hydration reminder.
- `RESTROOM`: light joke with a hard privacy rule: never speak the person's name.
- `WAVE`: semantic-event support in addition to the existing gesture path.

## Reliability / anti-spam

- stable event contract and parser;
- duplicate event-ID suppression;
- stale/future event rejection;
- per-person + per-event cooldown;
- global speech gap;
- same-person cross-event suppression;
- event priority and queue expiry;
- group arrival coalescing;
- identity enrichment from recent Global-ID bindings;
- generic fallback when identity is unknown;
- cache-first speech through the existing IMOU speaker path.

## Backward compatibility

If `/status.json` has no `audio_events` field, legacy presence greeting behaves as
before. When the detection team starts returning `"audio_events": []` (even when
empty), the bridge automatically switches to zone-event mode and disables
presence-only arrival greeting. This prevents duplicate greetings without forcing
both teams to merge on the same day.

## Detection-team integration

Read `backend/docs/AUDIO_EVENT_CONTRACT.md`.

Minimum preferred event:

```json
{
  "event_id": "camA-1727000123520-g27-door-enter",
  "type": "DOOR_ENTER",
  "global_id": 27,
  "person_id": "NV001",
  "display_name": "Minh",
  "camera": "A",
  "timestamp": 1727000123.520
}
```

The detection team should keep each new event in the `audio_events` status list
for at least 1-2 seconds. Repeated snapshots are safe because event IDs are
idempotently de-duplicated.

## Audio-only testing

From `backend`:

```powershell
python scripts/test_be_xinh_events.py --dry-run --event door_enter --name Minh
python scripts/test_be_xinh_events.py --dry-run --event door_exit --name Minh
python scripts/test_be_xinh_events.py --dry-run --event be_xinh_near --name Minh
python scripts/test_be_xinh_events.py --dry-run --event water --name Minh
python scripts/test_be_xinh_events.py --dry-run --event restroom --name Minh
```

For real IMOU playback:

```powershell
python scripts/prewarm_hamy.py
python scripts/test_be_xinh_events.py --event door_enter --name Minh --person-id NV001
```

## Verification performed

Focused audio/integration suite:

- 29 tests passed.

Full backend suite in the review environment:

- 277 tests passed;
- 1 pre-existing unrelated failure remains in `test_face_device.py` because
  `insightface` is not installed in the review sandbox, so that CUDA-provider test
  sees the dependency error before its expected `onnxruntime-gpu` error.

No detection or web code was modified to hide that failure.
