# Nhận diện trạng thái quan sát trên Camera Live

## Phạm vi

Module này chỉ mô tả trạng thái có thể quan sát từ camera. Nó không đánh giá năng suất,
cảm xúc, sức khỏe, ý định hay tính cách. `activity_status` tách biệt hoàn toàn với
`attendance_status` và `presence_status`.

## Kiến trúc

```text
RTSP frame
  -> YOLO11n-Pose (person box + COCO keypoints)
  -> CentroidTracker (camera_id + track_id)
  -> identity cache từ face recognition hiện có
  -> zone + pose + movement features
  -> temporal rules + rolling-window smoothing
  -> overlay/panel Camera Live
  -> activity_events (chỉ ghi khi transition)
```

- Face recognition và attendance không bị thay model hoặc thay luật.
- `ActivityRuntime` chạy trong worker riêng và bắt mọi lỗi pose. Worker attendance vẫn chạy
  nếu activity lỗi.
- Một `PoseService` được cache và dùng lại; model không được tạo lại theo frame.
- Camera vẫn dùng đúng RTSP sub-stream hiện có.
- Không lưu video hoặc ảnh mới.

## Model, tracker và luật phiên bản 1

- Person detector/pose: `yolo11n-pose.pt`, input mặc định 416 px.
- Tracker: centroid tracker nhẹ đã có trong project; state được tách theo camera.
- Identity: tâm face phải nằm trong person box; cache mặc định 15 giây.
- `AT_DESK`: tâm chân người nằm trong zone có ID thuộc `ACTIVITY_DESK_ZONE_IDS`, không có
  hoạt động ưu tiên cao hơn.
- `WALKING`: dịch chuyển net của track trong trail vượt ngưỡng chuẩn hóa theo đường chéo frame.
- `STANDING`: vai, hông, gối (và cổ chân nếu thấy) tạo cấu trúc đứng theo chiều dọc.
- `DRINKING`: cổ tay gần mũi, khuỷu tay nâng và duy trì tối thiểu 1.5 giây.
- `SLEEPING_SUSPECTED`: trong desk zone, đầu thấp gần/dưới đường vai, chuyển động rất thấp và
  duy trì tối thiểu 25 giây.
- `AWAY`: track đã từng ở desk zone, sau đó ra khỏi desk zone. Không thay thế ABSENT.
- `UNKNOWN`: thiếu bằng chứng; trạng thái ngắn không được ghi database.

Rolling window mặc định 5 giây, tỷ lệ xác nhận 70%. Thứ tự ưu tiên:
`DRINKING > SLEEPING_SUSPECTED > WALKING > STANDING > AT_DESK > AWAY > UNKNOWN`.

## Cấu hình

Các biến nằm trong `.env.example`, gồm bật/tắt module, model pose, frame skipping, kích thước
input, rolling window, tỷ lệ xác nhận, thời gian tối thiểu của từng trạng thái, movement
threshold, identity TTL và danh sách desk zone.

Zone dùng tọa độ chuẩn hóa trong `SPATIAL_ZONES_JSON`. Cần hiệu chỉnh polygon theo góc camera
thật trước khi dùng `AT_DESK`/`AWAY` trong vận hành.

## Lưu trữ

Table `activity_events` có các trường: employee, camera, track, activity, thời gian bắt đầu/kết
thúc, duration, confidence và zone. Event đang mở được đóng khi đổi trạng thái hoặc mất track.
Cùng một trạng thái liên tục chỉ cập nhật event mở, không tạo record theo từng frame.

## Giao diện

- Camera Live: person box, activity code, confidence; panel “Trạng thái hiện tại” hiển thị tên,
  trạng thái tiếng Việt và thời lượng.
- Báo cáo & Phân tích > Không gian & Ra vào > Hoạt động: biểu đồ phân bổ thời lượng quan sát và
  danh sách event. Không có score hay xếp hạng nhân viên.

## Benchmark ngày 23/09/2026

Đo read-only trên hai RTSP sub-stream bằng `scripts/benchmark_activity.py`; không gọi attendance.

| Camera | FPS camera trước AI | FPS camera khi AI chạy | Pose steady-state | Face detection hiện tại |
|---|---:|---:|---:|---:|
| Camera 1 | 19.54 | 14.96 | 34.3 ms / ~29.1 infer/s | 2535.4 ms/lần |
| Camera 2 | 25.12 | 15.07 | 26.6 ms / ~37.6 infer/s | 1970.0 ms/lần |

Activity thực tế mặc định chỉ xử lý mỗi 3 frame, mục tiêu khoảng 3–5 cập nhật/giây. Lần gọi pose
đầu tiên mất khoảng 2.4 giây để nạp model. Kết quả phụ thuộc CPU, số người, mạng RTSP và góc camera.
Nút thắt đo được là face model hiện hữu, không phải YOLO Pose.

Chạy lại benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_activity.py --camera 0
```

## Kiểm thử và hạn chế

- `tests/test_activity_recognition.py` bao phủ 16 case yêu cầu và kiểm tra AWAY tách khỏi attendance.
- Hai camera thật đều mở được; smoke test Camera 1 đạt 14.96 FPS capture và 4.93 activity FPS,
  worker chạy ổn định, không lỗi và phân biệt track độc lập.
- Heuristic uống nước ưu tiên false negative; có thể bỏ sót khi cổ tay/khuôn mặt bị che.
- `SLEEPING_SUSPECTED` chỉ là dấu hiệu tư thế, không phải kết luận người đang ngủ.
- Camera top-down, người ngồi bị bàn che, ánh sáng yếu và người giao nhau có thể làm pose/tracker
  sai. Phiên bản này chưa có re-identification xuyên camera.
- Accuracy chưa được benchmark trên dataset gán nhãn, vì vậy không công bố phần trăm chính xác.
