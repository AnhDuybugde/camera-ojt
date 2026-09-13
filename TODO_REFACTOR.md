# Refactor TODO — Identity, Tracking và Attendance

## Đọc nhanh: hệ thống này làm gì?

Repo này nhận hình ảnh từ hai camera, phát hiện người bằng YOLO, theo dõi
người trong từng camera, suy luận trạng thái vị trí và nhận diện nhân viên
bằng khuôn mặt ở camera điểm danh. Kết quả cuối cùng được dùng cho overlay,
dashboard và lưu vào Supabase.

Vấn đề lớn của thiết kế cũ là ba khái niệm ID bị dùng lẫn nhau. `local_track_id`
chỉ có ý nghĩa trong một camera và có thể đổi khi người bị che khuất. Vì vậy
ID này không được dùng làm danh tính lâu dài. `global_person_id` là kết quả
association tạm thời: nó đoán các track ở các camera khác nhau có thể là cùng
một người. Còn `employee_id` mới là danh tính bền vững lấy từ face gallery.
Global ID có thể bị tạo lại, gộp hoặc chuyển giao; việc đó không được làm mất
employee identity đã xác nhận.

Luồng production sau refactor được tổ chức như sau:

```text
RTSP (mỗi camera decode một lần)
        -> FrameHub/latest frame
        -> YOLO + ByteTrack (local track)
        -> GlobalIdentityManager (association tạm thời)
        -> TrackEvent
             |-> WorkstateConsumer
             |-> FaceTrackConsumer -> employee_id -> Attendance
             |-> renderer/dashboard
             |-> async storage queue -> Supabase
```

Các nhánh nghiệp vụ cùng đọc một `TrackEvent`, nên attendance không phải chờ
workstate, dashboard hoặc Supabase xử lý xong. Consumer chậm sẽ bỏ frame cũ
để ưu tiên hình mới; storage lỗi chỉ làm tăng hàng đợi cục bộ chứ không dừng
detector.

## Những thay đổi đã hoàn thành

- [x] Production path `run_workstate.py` dùng `ByteTrack` cho local tracking.
- [x] Hai camera dùng chung một `GlobalIdentityManager`.
- [x] Thêm metadata riêng cho ba lớp identity:
  - `local_track_id`: ID tạm trong một camera.
  - `global_person_id`: ID association xuyên track/camera.
  - `employee_id`: ID bền vững từ face recognition.
- [x] `ByteTrack` và `IoUTracker` giữ lại local track ID.
- [x] `GlobalIdentityManager` trả ra global ID riêng với local track ID.
- [x] Thêm khả năng bind `employee_id` vào Global Identity.
- [x] Giảm số lần gọi face recognition theo từng track:
  - Track đã nhận diện: cooldown 30 giây.
  - Track chưa nhận diện: retry sau 1 giây.
  - Track không thấy mặt cũng không gọi detector liên tục.
- [x] Theo dõi chất lượng face crop và ưu tiên candidate tốt hơn.
- [x] Dọn face state khi Global ID bị retire.
- [x] Bổ sung test cho việc phân biệt local/global/employee ID.
- [x] Sửa module IMOU để không tự mở camera khi import; camera chỉ mở trong runtime entry point.
- [x] Dùng chung cơ chế latest-frame/low-buffer trong production loop; mỗi camera được decode một lần.
- [x] Supabase/face/event writes chuyển qua `WriteQueueWorker`, không block inference loop.
- [x] Chạy test suite thành công: `84 passed`.

## Checklist triển khai và lý do của từng nhóm

### P0 — Kiến trúc runtime

Đây là phần dọn đường cho toàn hệ thống. `run_workstate.py` là entry point
production duy nhất cho hai camera và dùng ByteTrack cho local tracking.
IoUTracker vẫn được giữ để chạy demo hoặc unit test cũ, nhưng không còn được
phép cùng cập nhật Global ID, workstate hay attendance trong runtime production.

`TrackEvent` là hợp đồng dữ liệu chung giữa inference và các consumer. Nhờ đó
face, workstate, visualization và storage có thể thay đổi độc lập mà không
tạo thêm một pipeline tracking song song. Mọi ghi database hoặc upload ảnh
đều đi qua `WriteQueueWorker` và SQLite queue; đặc biệt inference loop không
gọi Supabase trực tiếp.

- [x] Chuẩn hóa `scripts/run_workstate.py` là production entry point duy nhất cho deployment hai camera.
- [x] Ghi rõ `IoUTracker` chỉ dành cho legacy single-camera demo/unit test.
- [x] Thêm immutable `TrackEvent` contract và synchronous `TrackEventBus` fan-out.
- [x] Tách `FaceTrackConsumer` và `WorkstateConsumer` nhận cùng contract track.
- [x] Supabase write đã tách khỏi inference; lỗi workstate B được cô lập để face attendance vẫn tiếp tục.
- [x] Đưa toàn bộ database/storage write sang `WriteQueueWorker`/SQLite queue bất đồng bộ.

### P1 — Global Identity

Global Identity hiện chỉ làm nhiệm vụ liên kết track, không đóng vai trò
nguồn chân lý về nhân viên. Appearance chính dùng embedding OSNet person-ReID,
được L2-normalize và lưu nhiều mẫu trong gallery. Histogram màu chỉ là
fallback khi OSNet không cài được hoặc không tải được weight; màu sắc không
còn quyết định chính vì phụ thuộc mạnh vào ánh sáng, góc nhìn và camera.

Khi face recognition xác nhận một `employee_id`, hệ thống bind identity đó
vào Global ID hiện tại và reconcile các Global ID cũ đã mất. Hai track đang
active vẫn được giữ riêng nếu chưa đủ bằng chứng, tránh gộp nhầm hai người
đang xuất hiện cùng lúc. Topology camera, thời gian và khoảng cách chỉ dùng
để lọc candidate, không thay thế face identity.

- [x] Tích hợp OSNet person-ReID dạng optional backend, có fallback histogram.
- [x] Giữ histogram màu chỉ như fallback, không còn là backend mặc định.
- [x] Chuẩn hóa L2 embedding trước khi lưu gallery/tính cosine similarity.
- [x] Gallery từ chối crop body quá nhỏ hoặc confidence thấp; face crop có quality gate riêng.
- [x] Có cơ chế merge/reconcile Global ID khi face xác nhận cùng `employee_id`; duplicate active được giữ riêng khi còn mơ hồ.
- [x] Kiểm tra source: production session chỉ khởi tạo một `GlobalIdentityManager` dùng chung cho A/B.
- [x] Có test cross-camera, track switch, lifecycle; duplicate active được chặn một-to-một bởi Hungarian assignment.

### P2 — Face Attendance

Attendance không chạy nhận diện khuôn mặt trên mọi frame. Consumer chỉ xử lý
track đủ ổn định, kiểm tra kích thước mặt, độ sắc nét và detector confidence,
sau đó giữ best-shot trong một cửa sổ ngắn. Track đã biết được cooldown;
track chưa biết chỉ retry theo chu kỳ. Cách này vừa giảm tải InsightFace vừa
tránh lưu một crop xấu hoặc ghi attendance lặp lại.

Khi match thành công, kết quả được gắn vào employee identity, Global ID,
workstate và attendance service. Attendance được debounce theo nhân viên và
ngày; ảnh chỉ lưu best-shot, còn database lưu path và metadata.

- [x] Có debounce attendance theo employee và theo ngày.
- [x] Có best-crop storage thay vì lưu mọi crop.
- [x] Có cooldown nhận diện theo track.
- [x] Tách face detection/matching thành `FaceTrackConsumer` nhận `TrackEvent`.
- [x] Face match bind vào manager metadata và `gid_to_person`/workstate/storage downstream.
- [x] Quality score tổng hợp kích thước + blur + detector confidence; pose/visibility được reject gián tiếp qua face bbox/crop gate.
- [x] Chỉ retry face recognition khi track mới, identity giảm confidence hoặc cooldown hết.
- [x] Bổ sung unit test face/workstate consumer với cooldown và Global ID.

### P3 — Performance và camera layer

Camera chỉ được mở và decode một lần. `FrameHub` giữ frame mới nhất thay vì
queue vô hạn, còn `LatestJpegRenderer` encode JPEG ở worker riêng để việc
streaming không chiếm thời gian inference. `StageMetrics` cho biết thời gian
của detection, tracking, face và rendering.

Detection FPS được tách khỏi FPS của camera bằng
`camera.process_every_n_frames`. Script benchmark dùng frame tổng hợp nên có
thể đo CPU/GPU mà không cần mở RTSP; con số này dùng để chọn chu kỳ xử lý phù
hợp cho máy triển khai.

- [x] Thêm `FrameHub` latest-frame abstraction tái sử dụng; không có unbounded queue.
- [x] Production loop đã dùng low-buffer RTSP và đọc mỗi camera một lần; frame cũ không được xếp queue inference.
- [x] Tách rendering/MJPEG khỏi inference loop bằng `LatestJpegRenderer` (latest-frame worker).
- [x] Đo thời gian các stage inference/rendering bằng `StageMetrics`; decode/storage vẫn được theo dõi qua log/queue và chưa có benchmark riêng.
- [x] Có `scripts/benchmark_inference.py` đo CPU/GPU độc lập RTSP; đã chạy xác nhận ở image size 320: CUDA `81.85 FPS`, CPU `17.04 FPS`.
- [x] `ResilientCapture` reconnect giữ nguyên manager/tracker session; chỉ reconnect capture.

### P4 — Workstate và dữ liệu

Workstate chỉ kết luận trạng thái vị trí quan sát được, chẳng hạn `Working`
(đang ở workstation), `Away`, `Returning` hoặc `Out of office`; movement
không đủ để chứng minh một người thực sự đang làm việc. State machine dùng
dwell time và hysteresis để tránh nhấp nháy do detector miss.

Database tách current state khỏi history event. Current state chỉ upsert khi
trạng thái thay đổi hoặc có heartbeat cần thiết; room status và current state
được coalesce theo key để không tạo hàng nghìn bản ghi giống nhau. Event
attendance/leave/return vẫn được giữ riêng để không mất lịch sử.

- [x] Workstation layer dùng `AT_WORKSTATION`/`AWAY_FROM_WORKSTATION`; label `Working` chỉ là presentation tương thích dashboard.
- [x] Giữ state machine dựa trên dwell time, hysteresis và transition rõ ràng; WorkstateConsumer chỉ làm adapter.
- [x] Thêm bảng employee-centric `employee_current_state`; room events vẫn lưu history.
- [x] Chỉ upsert current state khi state/room status thay đổi; attendance có debounce theo employee/ngày.
- [x] Attendance đã có unique constraint `(date, person_id)`.
- [x] Face crop được lưu local best-shot và upload object storage; database giữ path/metadata.

### Model và thuật toán chưa ưu tiên

Không đổi model chỉ để chữa lỗi orchestration. ByteTrack phù hợp với local
tracking, InsightFace phù hợp với face identity và OSNet bổ sung appearance
cho cross-camera association. BoT-SORT hoặc activity recognition chỉ nên
được benchmark/thêm sau khi có dữ liệu ID-switch hoặc requirement activity
thực tế; chúng không phải điều kiện cần của refactor này.

- [x] Đã kiểm tra OSNet forward thành công trên CPU với embedding 512 chiều.
- [x] Giữ ByteTrack; BoT-SORT được hoãn có chủ đích tới khi có dữ liệu ID-switch để benchmark.
- [x] Không thêm activity recognition/pose model: ngoài phạm vi state vị trí hiện tại.
- [x] Giữ InsightFace vì model phù hợp; đã sửa triggering, crop quality và orchestration.

## Kiểm thử cần chạy trước khi demo production

- [x] `python -m pytest -q` — `84 passed`.
- [x] Live smoke 10 frame, hai RTSP, YOLO CUDA, `--no-face`: pipeline khởi động/kết thúc bình thường.
- [x] Live smoke 10 frame, hai RTSP, YOLO CUDA, OSNet + InsightFace: attendance path khởi động bình thường.
- [x] Dashboard `npm run build` thành công.
- [x] Unit test hai camera cùng lúc với một Global Identity Manager.
- [x] Unit test camera A → camera B qua Global Identity; live chưa có người để xác nhận.
- [x] Unit test mất track ngắn và mất track dài.
- [x] Unit test face/employee binding sau khi Global ID đã đổi; live chưa có face match.
- [x] Smoke khi Supabase không ghi được: inference vẫn kết thúc, write còn trong SQLite queue (`19` pending sau smoke).
- [x] Unit test renderer/MJPEG publish độc lập với inference; kiểm thử end-to-end dashboard dừng vẫn để dành cho deployment.
- [x] Queue coalesce `current_state` theo employee và `room_status` theo ngày/global ID; event/attendance vẫn giữ nguyên để không mất lịch sử.

## Trạng thái hiện tại

Đây là trạng thái code sau refactor, không phải cam kết rằng mọi camera đều
nhận diện hoàn hảo trong mọi điều kiện ánh sáng. Unit test và smoke test chứng
minh pipeline khởi động, phân biệt ID, xử lý reconnect, queue offline và
consumer độc lập. Live smoke chưa có người xuất hiện trong cửa sổ kiểm thử,
nên việc match xuyên camera và attendance với người thật vẫn cần kiểm thử
nghiệm thu khi triển khai.

- Test suite hiện tại: `84 passed`.
- Đã hoàn thành phần tách identity và giảm face inference lặp.
- Đã có contract fan-out, `FrameHub`, `FaceTrackConsumer`, `WorkstateConsumer`, OSNet backend, async storage và renderer worker; attendance persistence/reconcile vẫn nằm ở orchestration layer vì cần transaction context của session.
- Live smoke chưa thấy người trong 10 frame, nên chưa chứng minh được cross-camera match/attendance thực tế.
