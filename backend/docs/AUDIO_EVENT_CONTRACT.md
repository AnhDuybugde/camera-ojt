# Bé Xinh Audio Event Contract v1

Detection/tracking owns camera frames, polygons, foot points, direction and
dwell detection. Audio owns phrases, privacy, cooldown, priority, queue expiry,
TTS cache and the IMOU speaker. Audio does not reopen RTSP or recalculate zones.

## Status payload

Zone Mode is enabled by a top-level list, including when it is empty:

```json
{"people": [], "audio_events": []}
```

When the field is absent, the bridge keeps the existing presence/gesture path.
When present, a face heartbeat alone cannot trigger an arrival greeting.

Preferred event:

```json
{
  "event_id": "camA-1727000123520-g27-door-enter",
  "type": "DOOR_ENTER",
  "global_id": 27,
  "person_id": "NV001",
  "display_name": "Minh",
  "camera": "A",
  "timestamp": 1727000123.520,
  "confidence": 0.94,
  "direction": "outside_to_inside"
}
```

Keep an event in `audio_events` for 1–2 seconds. Repeated snapshots are safe
when `event_id` remains stable.

## Supported semantic events

- `DOOR_ENTER`, `DOOR_EXIT`
- `BE_XINH_NEAR`
- `WATER`
- `RESTROOM`
- `WAVE`

Generic `ZONE_DWELL` is supported for `be_xinh`, `water`, `restroom`/`wc`.
These zones require the configured minimum dwell; `ZONE_ENTER` alone is
rejected. Door zones support `ZONE_ENTER`, `ZONE_EXIT` and direction
`outside_to_inside` / `inside_to_outside`.

## Privacy and reliability

- `RESTROOM` never renders a person's name in either the selected phrase or
  its fallback. Identity is used only for cooldown.
- Event IDs are de-duplicated with bounded TTL memory.
- Stale/future events, short dwell and queue-expired speech are rejected.
- Priority order: Wave, Door enter, Door exit, Near, Water, Restroom.
- Per-person/event cooldown, global gap and same-person cross-event gap prevent
  spam; simultaneous arrivals become one group greeting.

Test without IMOU:

```powershell
python scripts/test_be_xinh_events.py --dry-run --event door_enter --name Minh
python scripts/test_be_xinh_events.py --dry-run --generic-zone restroom --name Minh
```
