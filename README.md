# Camera Tracking

Pipeline Python theo doi luu luong nguoi, mat do va huong di chuyen trong phong tu
camera gan tren tuong. He thong dung YOLO26s tren device tu dong va chieu diem chan cua moi nguoi
tu anh camera xuong mat san bang homography.

## Sau stage

1. Foundation: config co validation, domain models va cac interface tach roi.
2. Camera input: webcam, video file hoac RTSP qua OpenCV.
3. Detection: YOLO26s, chi lay class `person`, lazy-load model.
4. Tracking: gan ID bang IoU tracker cho camera co dinh.
5. Floor analytics: trajectory theo met, occupancy, zone, density, heatmap, movement
   vector va dem vao/ra qua line.
6. Demo/output: CLI, video co overlay va JSON report.

Luongs du lieu chinh:

```text
FrameSource -> PersonDetector -> Tracker -> FloorProjector -> RoomAnalytics
                                                        -> Overlay + JSON/video
```

## Cai dat

Python 3.10 tro len:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Chay

Webcam mac dinh:

```powershell
python scripts\run_pipeline.py --config config\default.yaml --display
```

Video mau:

```powershell
python scripts\run_pipeline.py --source data\raw\room.mp4 --max-frames 300
```

## Chay tracking ID 2 camera

Lenh ngan 1 command 2 channel (RTSP tu `.env`, device auto cuda/mps/cpu):

```powershell
python scripts\run_workstate.py --display
```

Logic: moi channel chay YOLO -> ByteTrack (tracking ngan han). Mot
GlobalIdentityManager dung chung cho ca 2 channel giu Global Person ID on dinh
xuyen tracklet, xuyen mat dau dai va xuyen channel (appearance gallery +
cost matrix + Hungarian + gating + lifecycle ACTIVE/TEMP_LOST/LONG_LOST/UNRESOLVED).
Lop business theo channel (WORKING/AWAY_TEMP/POSSIBLY_OUT/RETURNING) chay rieng
va khong anh huong ID. Bo box nho hon `--min-area` de giam nhieu. Device `auto`
trong `config\default.yaml` (mac dinh cuda neu co GPU). Cau hinh identity nam
trong block `identity:` cua `config\default.yaml`.

Tuy chinh khi can:

```powershell
python scripts\run_workstate.py --source-a 0 --source-b data\samples\hallway.mp4 --display
python scripts\run_workstate.py --device cpu --display
python scripts\run_workstate.py --model yolo26s.pt --imgsz 960 --display
python scripts\run_workstate.py --min-area 5000 --display
```

Nhan `q` hoac ESC de thoat.

Nhan `q` de dong cua so khi dung `--display`. Ket qua mac dinh nam trong `output/`.
Lan chay dau, Ultralytics se tai `yolo26s.pt` khoang 20 MB; file weights duoc Git bo qua.

Neu OpenCV bao `The function is not implemented` tai `cvShowImage`, kiem tra va loai bo
cac ban OpenCV cai chong nhau:

```powershell
python -m pip uninstall -y opencv-python-headless opencv-contrib-python opencv-python
python -m pip install opencv-python==4.12.0.88
```

Moi Python environment chi nen co mot trong cac goi OpenCV tren. Neu khong can cua so,
bo `--display`; pipeline van ghi video `output/annotated.mp4`.

## Hieu chinh mat san

Bon `analytics.calibration.image_points` trong `config/default.yaml` la diem pixel tren
anh. Bon `floor_points` tuong ung la vi tri that tren san, tinh bang met va cung thu tu.
Cau hinh hien tai chi la vi du cho video 1280x720, can thay bang so do cua phong that.

He thong lay trung diem canh duoi bounding box lam diem tiep xuc voi san. Projection
nay can thiet voi camera gan tuong vi mot pixel o gan camera va mot pixel o xa camera
khong dai dien cho cung mot khoang cach that.

Xem huong dan tai [docs/CALIBRATION.md](docs/CALIBRATION.md).

## Test

```powershell
python -m pytest -q
```

Smoke test dung detector gia lap nen chay offline va kiem tra toan bo pipeline, bao gom
ID tracking, floor trajectory, line crossing, density/heatmap, overlay va JSON report.

## Cau truc

```text
config/                         runtime configuration
data/                           local videos and samples
docs/                           design and calibration notes
models/                         optional local model weights
output/                         generated videos and reports
scripts/run_pipeline.py         command-line entry point
src/camera_tracking/
  analytics/                    floor projection and room analytics
  camera/                       frame sources
  detection/                    detector interface and YOLO adapter
  tracking/                     multi-object ID tracking
  visualization/                video overlays
  config.py                     validated configuration
  domain.py                     shared data contracts
  output.py                     report and video writer
  pipeline.py                   stage orchestration
tests/                          unit and end-to-end smoke tests
```
