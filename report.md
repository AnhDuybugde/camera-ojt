# BÁO CÁO KẾT QUẢ DỰ ÁN CAMERA TRACKING

## 1. Thông tin chung

- **Tên dự án:** Camera Tracking
- **Repository:** `https://github.com/AnhDuybugde/camera-tracking`
- **Nhánh làm việc:** `main`
- **Ngôn ngữ chính:** Python
- **Bài toán:** Theo dõi lưu lượng người trong phòng, mật độ hiện tại, mật độ tích lũy
  và luồng di chuyển từ camera cố định gắn trên tường.
- **Đối tượng phát hiện:** Con người, class `person` của bộ dữ liệu COCO.
- **Môi trường mục tiêu:** Máy không có GPU hoặc có cấu hình thấp, xử lý bằng CPU.

## 2. Mục tiêu sản phẩm

Sản phẩm nhận video từ webcam, file video hoặc camera IP qua RTSP, sau đó thực hiện:

1. Phát hiện người xuất hiện trong từng khung hình.
2. Gán ID tạm thời cho từng người qua nhiều khung hình liên tiếp.
3. Ước lượng vị trí của mỗi người trên mặt sàn.
4. Lưu quỹ đạo di chuyển của từng ID trong hệ tọa độ mặt sàn.
5. Đếm số người đang có mặt trong khung hình.
6. Đếm số người trong từng vùng chức năng của phòng.
7. Tính mật độ hiện tại và heatmap tích lũy.
8. Xác định vector di chuyển gần nhất của từng người.
9. Đếm lượt vào và lượt ra khi một trajectory cắt counting line.
10. Hiển thị kết quả trực tiếp và xuất video, báo cáo JSON.

Sản phẩm hiện là một MVP chạy được end-to-end. Kết quả đo theo mét chỉ có ý nghĩa sau
khi camera thực tế được calibration đúng với mặt sàn của phòng.

## 3. Kiến trúc hệ thống

Hệ thống được chia thành sáu stage độc lập:

```text
Stage 1             Stage 2          Stage 3          Stage 4
Configuration  ->  Camera input  ->  Detection   ->  Tracking
                                                        |
                                                        v
Stage 6                                             Stage 5
Output/Display  <-  Visualization  <-  Room analytics + Floor projection
```

Luồng dữ liệu chi tiết:

```text
Webcam / Video / RTSP
        |
        v
OpenCVFrameSource
        | Frame(index, timestamp, image)
        v
YoloPersonDetector
        | list[Detection]
        v
IoUTracker
        | list[Track]
        v
FloorProjector
        | track_id + floor position (x, y)
        v
RoomAnalytics
        | AnalyticsSnapshot
        +--------------------+
        |                    |
        v                    v
OverlayRenderer         ResultWriter
        |                    |
        v                    +--> output/report.json
Live window                  +--> output/annotated.mp4
```

Các module trao đổi bằng những data contract chung, không gọi trực tiếp vào phần triển
khai nội bộ của nhau. Vì vậy có thể thay YOLO bằng detector khác hoặc thay IoU tracker
bằng ByteTrack mà không phải viết lại analytics.

## 4. Cấu trúc source code

```text
camera-ojt/
|-- config/
|   `-- default.yaml                 Cấu hình chạy mặc định
|-- data/
|   |-- raw/                         Video gốc
|   |-- processed/                   Dữ liệu trung gian
|   `-- samples/                     Dữ liệu kiểm thử nhỏ
|-- docs/
|   |-- CALIBRATION.md               Hướng dẫn hiệu chỉnh mặt sàn
|   `-- PROJECT_BRIEF.md             Mô tả bài toán
|-- models/                          Model lưu cục bộ nếu cần
|-- output/                          Video và báo cáo sinh ra
|-- scripts/
|   `-- run_pipeline.py              CLI khởi chạy sản phẩm
|-- src/camera_tracking/
|   |-- analytics/
|   |   |-- engine.py                Density, heatmap, zone, flow, counting
|   |   `-- projection.py            Homography ảnh sang mặt sàn
|   |-- camera/
|   |   `-- source.py                Webcam, video và RTSP input
|   |-- detection/
|   |   |-- base.py                  Interface PersonDetector
|   |   `-- yolo.py                  YOLO11 adapter
|   |-- tracking/
|   |   `-- iou.py                   IoU multi-object tracker
|   |-- visualization/
|   |   `-- overlay.py               Camera overlay và floor mini-map
|   |-- config.py                    Pydantic configuration models
|   |-- domain.py                    Shared data contracts
|   |-- output.py                    Ghi video và JSON
|   `-- pipeline.py                  Điều phối toàn bộ stage
|-- tests/
|   |-- test_config.py               Kiểm thử cấu hình
|   `-- test_smoke_pipeline.py       Smoke test end-to-end
|-- pyproject.toml                   Package metadata và dependencies
|-- requirements.txt                 Danh sách dependency
`-- README.md                        Hướng dẫn sử dụng nhanh
```

## 5. Mô tả cách code hoạt động

### 5.1. Configuration

`src/camera_tracking/config.py` định nghĩa cấu hình bằng Pydantic. Cấu hình được đọc từ
YAML và validate trước khi mở camera hoặc tải model.

Các nhóm cấu hình:

- `camera`: nguồn video, chiều rộng, chiều cao, FPS và tỷ lệ bỏ frame.
- `detection`: model, confidence, class ID, image size và device.
- `tracking`: IoU threshold, số hit xác nhận và thời gian giữ track bị mất.
- `analytics`: kích thước sàn, calibration, grid, zone và counting line.
- `output`: bật/tắt video, JSON và cửa sổ hiển thị.

Các trường không được khai báo sẽ bị từ chối nhờ `extra="forbid"`. Một calibration hợp
lệ phải có ít nhất bốn cặp điểm và hai danh sách phải có cùng số lượng phần tử.

### 5.2. Camera input

`OpenCVFrameSource` sử dụng `cv2.VideoCapture` và hỗ trợ:

- Số nguyên như `0`: webcam.
- Chuỗi đường dẫn: file video.
- URL có `://`: RTSP hoặc network stream.

Mỗi ảnh được đóng gói trong `Frame` gồm:

```python
Frame(index=frame_number, timestamp_s=time_in_seconds, image=numpy_array)
```

`process_every_n_frames: 2` nghĩa là pipeline chỉ xử lý một trong mỗi hai frame. Với
camera 25 FPS, tốc độ đầu vào analytics mục tiêu là khoảng 12.5 frame/giây. Cách này
giảm tải CPU nhưng chuyển động giữa hai frame sẽ lớn hơn.

### 5.3. Person detection

`YoloPersonDetector` triển khai interface `PersonDetector`. Model mặc định là
`yolo11n.pt`, phiên bản nhỏ nhất của YOLO11 detection.

Model được lazy-load khi frame đầu tiên tới. Lần chạy đầu Ultralytics tự tải weights;
những lần sau dùng file đã có trên máy. Lệnh inference chỉ yêu cầu class `person`:

```text
image -> YOLO11n -> bounding box + confidence + class_id
```

Kết quả của Ultralytics được đổi sang `Detection` nội bộ. Nhờ vậy các stage phía sau
không phụ thuộc vào kiểu dữ liệu riêng của Ultralytics.

### 5.4. Tracking

`IoUTracker` so sánh bounding box mới với trạng thái track cũ bằng Intersection over
Union:

```text
IoU = diện tích phần giao / diện tích phần hợp
```

Các cặp detection-track được sắp xếp theo IoU giảm dần. Một cặp được ghép nếu:

- IoU lớn hơn hoặc bằng `iou_threshold`.
- Detection chưa được ghép với track khác.
- Track chưa được ghép với detection khác.

Detection không được ghép sẽ tạo ID mới. Track chỉ được coi là tin cậy sau `min_hits`
lần phát hiện. Track mất detection được giữ tối đa `max_lost_frames` để có thể nối lại.

Ưu điểm của IoU tracker là nhẹ, dễ đọc và không thêm dependency. Hạn chế là có thể đổi
ID khi người di chuyển nhanh, bị che khuất hoặc đứng gần nhau.

### 5.5. Điểm tiếp xúc mặt sàn

YOLO11n detection không nhận diện riêng bàn chân. Code dùng trung điểm cạnh dưới của
bounding box làm ground-contact point:

```text
(x1, y1) +-------------+
         |             |
         |   person    |
         |             |
         +------o------+ (x2, y2)
                |
                +-- foot_point = ((x1 + x2) / 2, y2)
```

Đây là phép xấp xỉ phổ biến khi toàn thân người nhìn thấy được. Nếu chân bị bàn, ghế,
người khác che hoặc bounding box bị cắt bởi cạnh ảnh, ground point có thể sai.

### 5.6. Floor-plane projection

`FloorProjector` nhận các cặp điểm tương ứng:

```text
image_points: tọa độ pixel trong frame camera
floor_points: tọa độ thực tương ứng trên sàn, đơn vị mét
```

OpenCV tính ma trận homography `H`. Mỗi foot point `p` được chiếu xuống mặt sàn:

```text
p_floor ~ H * p_image
```

Homography xử lý biến dạng phối cảnh của camera gắn tường. Hai người đứng trên cùng
mặt sàn có thể xuất hiện ở các độ cao khác nhau trong ảnh, nhưng ground point của họ
vẫn được ánh xạ về vị trí tương ứng trên mặt sàn.

Điều kiện để phép chiếu có ý nghĩa:

- Sàn gần phẳng và các điểm calibration cùng nằm trên mặt sàn đó.
- Camera cố định vị trí, góc quay và zoom.
- Các cặp image/floor point được đo chính xác và cùng thứ tự.
- Ground point của người không bị che hoặc cắt.

Các điểm trong `config/default.yaml` chỉ là ví dụ cho ảnh 1280x720 và phòng 8 x 6 m,
không phải kết quả calibration của phòng thực tế.

### 5.7. Trajectory và movement vector

Với mỗi track đã xác nhận, `RoomAnalytics` thực hiện:

```python
floor_position = projector.project(track.bbox.foot_point)
trajectory[track_id].append(floor_position)
```

Trajectory không được set cứng. Nó được tạo liên tục từ dữ liệu của camera. Mỗi ID giữ
tối đa `trajectory_length` điểm, mặc định là 120.

Movement vector gần nhất được tính bằng:

```text
movement = current_floor_position - previous_floor_position
```

Vector hiện biểu diễn độ dịch chuyển giữa hai frame được xử lý. Muốn tính vận tốc theo
m/s cần chia độ dịch chuyển cho chênh lệch timestamp và thêm bước smoothing.

### 5.8. Occupancy, zone và density

`occupancy` là số track đã xác nhận trong frame hiện tại.

Mỗi zone là một polygon trong hệ tọa độ mặt sàn. OpenCV `pointPolygonTest` kiểm tra vị
trí mỗi người nằm trong zone nào để tạo `zone_counts`.

Mặt sàn được chia thành grid. Với điểm `(x, y)`, cell được xác định gần tương đương:

```text
column = floor(x / floor_width * number_of_columns)
row    = floor(y / floor_height * number_of_rows)
```

Hệ thống duy trì hai loại grid:

- `density_grid`: số người trong từng ô ở frame hiện tại.
- `heatmap_grid`: tổng số lần các vị trí xuất hiện trong từng ô từ đầu phiên chạy.

Heatmap hiện đo số sample theo frame, chưa chuẩn hóa thành người/phút hoặc người/m².

### 5.9. Đếm vào và ra

Counting line là một đoạn thẳng trên mặt sàn. Code dùng tích có hướng để xác định vị
trí của track nằm ở phía nào của line.

Một lượt crossing chỉ được ghi nhận khi:

- Track đổi từ một phía sang phía còn lại.
- Đoạn di chuyển thực sự cắt đoạn counting line, không chỉ cắt đường thẳng kéo dài.
- Hướng cắt khớp `entry_direction` thì tăng `entries`; hướng ngược lại tăng `exits`.

Nếu một sample nằm chính xác trên line, code giữ phía hợp lệ trước đó và chờ sample
tiếp theo, tránh bỏ mất crossing.

### 5.10. Visualization

`OverlayRenderer` tạo một bản sao của frame và vẽ:

- Bounding box cho track đã xác nhận.
- Track ID và confidence.
- Camera-view trajectory được biến đổi ngược từ mặt sàn về ảnh.
- Occupancy, số lượt vào và số lượt ra.
- Floor mini-map ở góc ảnh.
- Heatmap tích lũy, zone, counting line và trajectory trên mini-map.

Khi dùng `--display`, pipeline gọi HighGUI để mở cửa sổ. Trước khi chạy, code kiểm tra
OpenCV có GUI support hay không. Nếu đang dùng bản headless, chương trình báo lỗi rõ
ràng và không gọi cleanup gây che mất lỗi gốc.

### 5.11. Output

`ResultWriter` hỗ trợ hai sản phẩm đầu ra:

1. `output/annotated.mp4`: video đã vẽ detection, ID và analytics.
2. `output/report.json`: snapshot analytics cuối phiên.

Các trường chính trong JSON:

```json
{
  "last_frame": 100,
  "duration_s": 8.0,
  "occupancy": 3,
  "entries": 7,
  "exits": 4,
  "density_grid": [[0, 1], [2, 0]],
  "heatmap_grid": [[10, 38], [52, 16]],
  "zone_counts": {"entrance": 1, "room_center": 2},
  "floor_positions_m": {"1": [3.2, 1.8]},
  "trajectories_m": {"1": [[3.0, 1.7], [3.2, 1.8]]},
  "movement_vectors_m": {"1": [0.2, 0.1]}
}
```

JSON trên chỉ minh họa cấu trúc, không phải số liệu của một lần đo thực tế.

## 6. Cấu hình mặc định

Các thiết lập chính trong `config/default.yaml`:

| Thành phần | Giá trị mặc định | Ý nghĩa |
|---|---:|---|
| Camera source | `0` | Webcam mặc định |
| Frame size | `1280 x 720` | Kích thước yêu cầu từ camera |
| Camera FPS | `25` | FPS yêu cầu |
| Process interval | `2` | Xử lý một trong mỗi hai frame |
| Model | `yolo11n.pt` | YOLO11 nano |
| Confidence | `0.35` | Ngưỡng detection |
| Image size | `640` | Kích thước inference |
| Device | `cpu` | Không yêu cầu GPU |
| IoU threshold | `0.3` | Ngưỡng ghép track |
| Minimum hits | `2` | Số detection để xác nhận track |
| Maximum lost frames | `15` | Thời gian giữ track bị mất |
| Floor size | `8 x 6 m` | Kích thước ví dụ |
| Density grid | `4 x 6` | Số hàng và cột |
| Trajectory length | `120` | Số điểm tối đa mỗi ID |

## 7. Cài đặt và chạy sản phẩm

### 7.1. Tạo môi trường

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Hoặc cài bằng requirements:

```powershell
python -m pip install -r requirements.txt
```

### 7.2. Chạy webcam

```powershell
python scripts\run_pipeline.py --config config\default.yaml --display
```

Nhấn `q` để dừng cửa sổ.

### 7.3. Chạy file video

```powershell
python scripts\run_pipeline.py --source data\raw\room.mp4 --display
```

### 7.4. Chạy giới hạn frame

```powershell
python scripts\run_pipeline.py --source data\raw\room.mp4 --max-frames 300
```

### 7.5. Chạy không có GUI

```powershell
python scripts\run_pipeline.py --source data\raw\room.mp4
```

Không dùng `--display` thì pipeline vẫn ghi `output/annotated.mp4` và
`output/report.json`.

## 8. Kết quả kiểm thử

### 8.1. Automated smoke test

Smoke test tạo bốn frame ảnh giả và một detector giả lập trả về một người đang di
chuyển qua counting line. Test kiểm tra toàn bộ luồng:

```text
Synthetic frame -> Detection -> Tracking -> Homography -> Analytics
                -> Overlay -> JSON writer
```

Các điều kiện đã kiểm tra:

- Config YAML được load và validate.
- Một người giữ cùng `track_id` qua bốn frame.
- Occupancy cuối phiên bằng 1.
- Zone count bằng 1.
- Trajectory có đủ bốn điểm.
- Movement vector đúng hướng.
- Heatmap nhận đủ bốn sample.
- Điểm đi chính xác qua counting line vẫn được tính.
- `entries` tăng 1 và `exits` không tăng.
- JSON report được tạo và có trajectory.
- OpenCV headless được nhận diện đúng.

Kết quả gần nhất:

```text
Ran 3 tests
OK
```

### 8.2. Compile test

Toàn bộ `src`, `scripts` và `tests` đã chạy qua `compileall` thành công, không có lỗi
syntax.

### 8.3. YOLO11n CPU smoke test

YOLO11n thật đã được tải và chạy inference trên ảnh rỗng bằng CPU. Ảnh rỗng trả về 0
detection đúng như mong đợi.

Kết quả đo trên máy phát triển, ảnh đầu vào 1280x720 và `imgsz=640`:

```text
Frame đầu, gồm load model: khoảng 9.711 giây
Frame warm-up tiếp theo:    khoảng 0.126 giây
Frame warm-up tiếp theo:    khoảng 0.123 giây
Tốc độ ổn định quan sát:    khoảng 8.1 FPS
```

Con số này phụ thuộc CPU, nguồn video và số người trong ảnh. Thời gian frame đầu không
đại diện cho tốc độ vận hành vì bao gồm import Torch và nạp weights.

### 8.4. Full pipeline với webcam và GUI

Pipeline thật đã chạy một frame với webcam, YOLO11n, tracking, analytics, overlay, GUI
và report writer. Lệnh kiểm tra:

```powershell
python scripts\run_pipeline.py --config config\default.yaml --display --max-frames 1
```

Kết quả:

```text
Done: occupancy=0, entries=0, exits=0, report=output\report.json
```

OpenCV `imshow` cũng đã được kiểm tra mở và đóng cửa sổ thành công.

## 9. Sự cố OpenCV đã xử lý

Môi trường ban đầu có ba package ghi đè chung namespace `cv2`:

```text
opencv-contrib-python 4.10.0.84
opencv-python 4.12.0.88
opencv-python-headless 4.10.0.84
```

Module được import có `GUI: NONE`, gây lỗi tại `cvShowImage` và
`cvDestroyAllWindows`. Môi trường đã được sửa để chỉ còn:

```text
opencv-python 4.12.0.88
GUI: WIN32UI
```

Nếu lỗi tái diễn trong môi trường Python khác, chạy:

```powershell
python -m pip uninstall -y opencv-python-headless opencv-contrib-python opencv-python
python -m pip install opencv-python==4.12.0.88
```

Một Python environment chỉ nên có một biến thể OpenCV phù hợp nhu cầu.

## 10. Phiên bản thư viện đã xác minh

| Thư viện | Phiên bản trong môi trường kiểm thử |
|---|---:|
| Python | `3.13.7` |
| NumPy | `2.2.5` |
| OpenCV | `4.12.0.88` |
| Pydantic | `2.13.4` |
| PyYAML | `6.0.2` |
| Ultralytics | `8.3.237` |

Project khai báo hỗ trợ Python từ 3.10 trở lên. Khi triển khai mới, Python 3.11 hoặc
3.12 thường là lựa chọn bảo thủ hơn cho hệ sinh thái computer vision, dù pipeline đã
chạy thành công trên Python 3.13.7 tại máy phát triển.

## 11. Đánh giá kết quả sản phẩm

### 11.1. Phần đã hoàn thành

- Kiến trúc module rõ ràng, dependency đi theo một chiều.
- Chạy được webcam, video và có cấu trúc hỗ trợ RTSP.
- YOLO11n chạy CPU và chỉ phát hiện người.
- Tracking có ID và cơ chế xác nhận/mất track.
- Có phép chiếu homography xuống mặt sàn.
- Có trajectory, movement vector, occupancy, zone, density và heatmap.
- Có counting line theo hướng vào/ra.
- Có camera overlay và floor mini-map.
- Có video output và JSON report.
- Có validation cấu hình và automated smoke test.
- Có xử lý trường hợp OpenCV không hỗ trợ GUI.

### 11.2. Phần chưa được xác minh bằng dữ liệu thực tế

- Độ chính xác detection trong phòng mục tiêu.
- Độ chính xác calibration theo mét.
- Sai số trajectory khi chân bị che.
- Tỷ lệ đổi ID trong cảnh đông người.
- Độ chính xác đếm vào/ra trong nhiều giờ liên tục.
- Độ ổn định và reconnect của RTSP khi mất mạng.
- Hiệu năng với độ phân giải và số camera triển khai thật.

Do chưa có video calibration và số đo ground truth của phòng, không nên công bố phần
trăm accuracy cho floor position, counting hoặc occupancy ở thời điểm hiện tại.

## 12. Giới hạn kỹ thuật

### 12.1. Ground point chỉ là xấp xỉ

Bottom-center của bounding box không phải ankle keypoint. Sai số tăng khi chân bị che,
bounding box không bao phủ toàn thân hoặc người đứng sát cạnh ảnh.

### 12.2. Sàn phải gần phẳng

Một homography chỉ mô tả tốt một mặt phẳng. Người đứng trên bục, cầu thang hoặc tầng
cao khác sẽ bị chiếu sai. Trường hợp nhiều mặt phẳng cần nhiều calibration hoặc mô hình
camera 3D.

### 12.3. Tracker chưa tối ưu cho che khuất

IoU tracker không có appearance embedding và motion model đầy đủ. Hai người giao nhau
có thể đổi ID, làm trajectory bị nối sai.

### 12.4. Heatmap phụ thuộc sampling rate

Heatmap hiện cộng một sample cho mỗi track ở mỗi frame xử lý. Nếu thay đổi
`process_every_n_frames`, giá trị tuyệt đối của heatmap cũng thay đổi. Cần chuẩn hóa theo
thời gian để so sánh giữa các phiên có FPS khác nhau.

### 12.5. Counting chưa chống dao động quanh line

Nếu vị trí rung qua lại quanh counting line, cùng một người có thể tạo nhiều crossing.
Cần thêm hysteresis, vùng đệm hoặc trạng thái cooldown cho triển khai thực tế.

### 12.6. Chưa lưu lịch sử dạng sự kiện

JSON hiện là snapshot cuối phiên cùng trajectory còn trong bộ nhớ. Hệ thống chưa ghi
event streaming, CSV theo thời gian hoặc database.

## 13. Đề xuất nâng cấp

Thứ tự nâng cấp đề xuất:

1. Thu một video phòng thật có số người và hướng đi đã biết.
2. Xây dựng công cụ click bốn hoặc nhiều điểm để calibration trực quan.
3. Đánh giá sai số floor position bằng các mốc đo trên sàn.
4. Thay IoU tracker bằng ByteTrack hoặc BoT-SORT qua cùng interface.
5. Thêm smoothing cho trajectory và loại điểm nhảy bất thường.
6. Thêm hysteresis/cooldown cho counting line.
7. Thêm YOLO11 Pose để lấy ankle keypoint khi chân nhìn thấy rõ.
8. Dùng bottom-center làm fallback khi pose confidence thấp.
9. Chuẩn hóa heatmap theo thời gian và diện tích, ví dụ người-phút/m².
10. Ghi event vào SQLite/PostgreSQL và xây API/dashboard.
11. Thêm reconnect, timeout và health check cho RTSP.
12. Export YOLO11n sang ONNX/OpenVINO để tối ưu CPU nếu cần.

## 14. Tiêu chí nghiệm thu đề xuất

Trước khi dùng trong thực tế, nên đo trên một tập video đại diện và đặt tiêu chí cụ thể:

| Chỉ số | Cách đo đề xuất |
|---|---|
| Occupancy accuracy | So số người dự đoán với nhãn theo từng giây |
| Entry/exit accuracy | So tổng crossing với người kiểm đếm thủ công |
| ID switch | Đếm số lần một người bị đổi ID |
| Floor-position error | Khoảng cách Euclidean đến mốc thật, đơn vị mét |
| Processing FPS | Trung bình và percentile trong phiên dài |
| RTSP stability | Thời gian chạy và số lần reconnect |

Ngưỡng nghiệm thu phải được thống nhất theo mục đích sử dụng. Bài toán thống kê mật độ
thường chấp nhận sai số lớn hơn bài toán an toàn hoặc kiểm soát ra vào.

## 15. Trạng thái Git tại thời điểm báo cáo

- Repository đã được khởi tạo và liên kết với remote.
- Nhánh hiện tại là `main`.
- Commit remote gần nhất: `f7ba34a Initial camera tracking project structure`.
- Phần triển khai pipeline, test và tài liệu trong báo cáo này hiện đang ở working tree,
  chưa được commit và chưa được push lên remote.

Weights `*.pt`, video đầu vào và thư mục `output/` được `.gitignore` loại khỏi Git để
tránh đưa file lớn hoặc dữ liệu runtime lên repository.

## 16. Kết luận

Dự án đã có một pipeline camera tracking hoàn chỉnh ở mức MVP, chạy bằng YOLO11n trên
CPU và đi qua đầy đủ sáu stage từ camera input đến báo cáo đầu ra. Thiết kế hiện tại ưu
tiên tính module, dễ đọc và dễ thay thế từng thành phần.

Floor-plane trajectory đã được đưa vào như thành phần trung tâm của analytics, không
phải dữ liệu set cứng. Tuy nhiên độ chính xác phụ thuộc trực tiếp vào calibration và
khả năng ước lượng điểm tiếp xúc chân. Bước quan trọng tiếp theo không phải thêm nhiều
tính năng, mà là thu video thực tế, calibration đúng phòng và đánh giá định lượng sai
số trước khi lựa chọn tracker hoặc pose model phức tạp hơn.
