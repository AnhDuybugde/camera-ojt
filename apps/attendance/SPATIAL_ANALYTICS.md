# Spatial Analytics extension

This is an isolated copy of the attendance project. It does not write to the
original project directory or its attendance database.

## Features

- Business flow dashboard with peak check-in/check-out time buckets.
- Attendance occupancy curve compared with camera-derived density.
- Four views: Dòng người, Hiện diện, Không gian, and Bất thường.
- Rule-based operational insights without duplicating late/early statistics.
- Person detection with YOLO11s; OpenCV HOG is used as a fallback.
- Stable per-camera track IDs and movement trails.
- Optional employee identity binding using the existing face embeddings.
- Normalized camera zones, live occupancy, movement transitions, and dwell time.
- Historical density timeline and zone/hour heatmap.
- Separate storage at `data/spatial_analytics.db`; raw video is not stored.

## Setup and run

```powershell
cd "D:\AI Mind\Bài toán điểm danh - Spatial Analytics"
.\setup.ps1
.\run.ps1
```

Sign in as an administrator and open **Phân tích không gian**. Click
**Bắt đầu phân tích** to start collecting data.

## Camera-zone calibration

The default layout divides each frame into three vertical demo zones. For the
real camera view, set `SPATIAL_ZONES_JSON` in `.env`. Coordinates are normalized
from `0` to `1`, so the same configuration works at different resolutions.

```dotenv
SPATIAL_ZONES_JSON=[{"id":"entrance","label":"Lối vào","points":[[0,0],[0.3,0],[0.3,1],[0,1]]},{"id":"workspace","label":"Khu làm việc","points":[[0.3,0],[1,0],[1,1],[0.3,1]]}]
```

Use one polygon per physical area visible to a camera. Cross-camera identity
handoff is intentionally not inferred: events remain scoped to their source
camera to avoid false movement paths.
