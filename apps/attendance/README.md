# Hệ thống điểm danh bằng nhận diện khuôn mặt

Ứng dụng Windows/Python nhận diện nhân viên từ webcam hoặc RTSP, ghi check-in/check-out vào SQLite và đồng bộ bất đồng bộ lên Google Sheets. SQLite luôn là nguồn dữ liệu chính: mất mạng hoặc lỗi Google API không làm dừng camera hay mất lượt điểm danh.

## Tính năng

- CRUD nhân viên trên giao diện Streamlit.
- Đăng ký 12 mẫu khuôn mặt, kiểm tra đúng một khuôn mặt, kích thước và độ nét; gom mẫu nhiễu và lưu vector chuẩn hóa trong SQLite BLOB.
- InsightFace `buffalo_l` (ArcFace) chạy qua ONNX Runtime CPU; không cần tự train model.
- Nhận diện cosine similarity với threshold cấu hình được, xử lý mỗi N frame và tái sử dụng kết quả giữa các frame.
- Camera chạy ở thread riêng, worker nhận diện riêng; webcam/RTSP tự thử kết nối lại.
- Check-in đầu ngày, check-out sau khoảng tối thiểu, cooldown và unique constraint chống trùng.
- Trạng thái `ON_TIME`/`LATE`, dashboard, lịch sử có bộ lọc/CSV và biểu đồ.
- Upsert Google Sheets theo khóa `(Employee ID, Date)`, retry nền cho các bản ghi `PENDING`/`ERROR`.
- Module liveness độc lập, mặc định tắt và fail-closed nếu bật khi chưa tích hợp chiến lược.
- Unit test cho vector, rule điểm danh, database, chống duplicate và mock sync.

## Kiến trúc

```mermaid
flowchart TD
    A[Webcam / RTSP Camera] --> B[Capture Thread]
    B --> C[InsightFace Detection + Alignment]
    C --> D[ArcFace Embedding]
    D --> E[Cosine Face Matching]
    E --> F[Attendance Service]
    F --> G[(SQLite - source of truth)]
    G --> H[Background Sync Worker]
    H --> I[Google Sheets - reporting mirror]
    G --> J[Streamlit Dashboard]
```

### Kết nối với backend camera-ojt

Khi `CAMERA_OJT_URL` được cấu hình (mặc định `http://127.0.0.1:8765`),
`camera-ojt` là nguồn realtime duy nhất: Streamlit hiển thị MJPEG và đọc
`camera-ojt.status.v1`, không mở RTSP hoặc chạy nhận diện khuôn mặt lần hai.
Worker bridge trong Streamlit tự chuyển `aimind_tick` từ
`../output/queue.db` vào SQLite. Nếu `person_id` camera trùng `employee_id`,
không cần khai báo mapping; các mã cũ khác nhau vẫn map qua
`data/person_map.json`.

Trang **Hệ thống AI & Trợ lý** dành cho quản trị viên hiển thị tình trạng
camera, bridge, voice, Gemini/Tavily và cho phép hỏi Hà Linh bằng văn bản.
Gemini/Tavily chỉ chạy ở backend nên API key không được gửi xuống trình duyệt.

Chạy từ repo root bằng `camera-ojt run` (backend `:8767` + pipeline `:8765`
+ UI `:8501` trong một lệnh). Không chạy `streamlit run app.py` riêng lẻ —
UI cần backend `:8767` đang nghe (biến `CAMERA_API_URL`).

Các lớp chính:

- `CameraManager`: chỉ giữ frame mới nhất để RTSP không tích tụ độ trễ.
- `FaceDetector`: lazy-load InsightFace khi lần đầu cần camera.
- `FaceRecognizer`: cache ma trận embedding và so khớp vector hóa.
- `AttendanceService`: transaction SQLite nguyên tử cho một nhân viên/một ngày.
- `SyncWorker`: thử đồng bộ định kỳ mà không chặn luồng nhận diện.
- `frontend/src/pages/*`: mỗi màn hình là một module riêng.

## Cài đặt trên Windows

Khuyến nghị Python 3.10, 3.11 hoặc 3.12 64-bit. Từ PowerShell tại thư mục project:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Hoặc dùng script đã xử lý sẵn chế độ UTF-8 cho đường dẫn tiếng Việt:

```powershell
.\setup.ps1
.\run.ps1
```

Project kèm wheel InsightFace trong `vendor/`. Wheel này giữ nguyên FaceAnalysis,
ArcFace và ONNX nhưng bỏ extension dựng mesh/mask không dùng đến, vì extension đó
yêu cầu Microsoft Visual C++ Build Tools khi cài trên Windows.

Nếu PowerShell chặn activate, có thể dùng trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Database và các thư mục `data/faces`, `logs` được tạo tự động ở lần chạy đầu. Không cần chạy migration thủ công.

### Model InsightFace

Lần đầu mở Register Face hoặc Live Attendance, InsightFace sẽ tải model `buffalo_l` vào cache người dùng. Máy cần Internet ở lần đầu; các lần sau chạy offline. Ứng dụng cấu hình `CPUExecutionProvider`, phù hợp laptop không có GPU. Việc tải model có thể mất vài phút.

Nếu môi trường doanh nghiệp chặn tải tự động, tải bộ model InsightFace hợp lệ trên một máy có mạng và chép thư mục model cache tương ứng sang máy chạy ứng dụng.

## Cấu hình `.env`

Copy `.env.example` thành `.env`. Những giá trị quan trọng:

```dotenv
CAMERA_SOURCE=0
FACE_THRESHOLD=0.50
PROCESS_EVERY_N_FRAMES=3
ATTENDANCE_COOLDOWN=60
CHECKOUT_MIN_MINUTES=240
WORK_START_TIME=08:00
LATE_THRESHOLD=08:15
GOOGLE_SHEET_ID=
GOOGLE_CREDENTIALS_PATH=
```

- Webcam thường là `0`; thử `1` nếu máy có camera thứ hai.
- RTSP: `CAMERA_SOURCE=rtsp://username:password@192.168.1.20:554/path`.
- Camera Imou nhiều kênh: đặt `CAMERA_CONFIG_FILE` trỏ tới file Python có các biến
  `USERNAME`, `RAW_PASSWORD`, `IP`, `PORT`. Ứng dụng chỉ parse tĩnh bốn giá trị,
  không thực thi code trong file, rồi tạo channel 1 và channel 2.
- `REGISTRATION_CAMERA_SOURCE=0` giữ webcam laptop cho màn hình đăng ký, trong khi
  Live Attendance dùng các luồng RTSP thực tế.
- Với mật khẩu có ký tự đặc biệt, URL-encode các ký tự đó. Không commit `.env`.
- `FACE_THRESHOLD` cao hơn giảm nhận nhầm nhưng tăng bỏ sót. Bắt đầu ở `0.50`, kiểm thử với camera/ánh sáng thật rồi hiệu chỉnh.
- `CHECKOUT_MIN_MINUTES=240` nghĩa là lần xuất hiện hợp lệ từ 4 giờ sau check-in trở đi sẽ là check-out.
- Sửa `.env` rồi restart Streamlit để áp dụng.

## Thiết lập Google Sheets

Google Sheets là tùy chọn; để trống hai biến Google thì hệ thống vẫn hoạt động bằng SQLite.

1. Mở Google Cloud Console và tạo một project.
2. Trong **APIs & Services**, enable **Google Sheets API** và **Google Drive API**.
3. Tạo **Service Account**.
4. Tạo key JSON cho Service Account và tải về, ví dụ đặt ngoài Git hoặc đặt tên `credentials.json` (đã được `.gitignore`).
5. Tạo Google Spreadsheet, bấm Share và cấp quyền Editor cho email `client_email` trong file JSON.
6. Lấy Spreadsheet ID trong URL: phần nằm giữa `/d/` và `/edit`.
7. Điền `.env`:

```dotenv
GOOGLE_SHEET_ID=your_spreadsheet_id
GOOGLE_CREDENTIALS_PATH=C:/secure/location/credentials.json
GOOGLE_WORKSHEET=Attendance
```

Worksheet `Attendance` và header sẽ tự được tạo. Worker nền thử lại mỗi 30 giây; màn hình **Google Sheets Sync** cho phép sync ngay và xem hàng chưa sync.

## Chạy demo

Tạo dữ liệu mẫu (không tạo embedding giả):

```powershell
python scripts/create_demo_data.py
streamlit run app.py
```

Luồng demo thực tế:

1. Vào **Employees**, thêm `NV001 - Nguyen Van A` nếu chưa chạy seed.
2. Vào **Register Face**, chọn `NV001`, nhìn thẳng rồi từ từ đổi góc mặt đến đủ mẫu.
3. Vào **Live Attendance**, bấm **Start camera**.
4. Khi nhận đúng người và vượt threshold, hệ thống ghi check-in, cập nhật dashboard/SQLite và xếp sync nền.
5. Sau `CHECKOUT_MIN_MINUTES`, lần nhận diện lại sẽ ghi check-out. Trong thời gian cooldown không tạo bản ghi lặp.
6. Xem/lọc/xuất dữ liệu tại **Attendance History**.

Database mặc định: `data/database.db`. Log xoay vòng: `logs/app.log`.

## Kiểm thử

Không cần camera, model hay Google credentials để chạy unit test:

```powershell
python -m pytest
python -m compileall -q .
```

## Xử lý lỗi thường gặp

- **Cannot open camera**: đóng Teams/Zoom, kiểm tra quyền Camera của Windows, đổi `CAMERA_SOURCE=1`, rồi restart app.
- **RTSP bị ngắt**: app tự reconnect. Kiểm tra URL, cùng mạng/VPN, firewall, codec và giới hạn số client của camera. Xem `logs/app.log`.
- **Model tải lỗi**: kiểm tra Internet/proxy và quyền ghi cache. Khởi động lại sau khi model đã được chép đủ.
- **Không detect mặt**: tăng ánh sáng, nhìn gần camera, giảm `MIN_FACE_SIZE`; không đăng ký khi có hơn một người trong khung.
- **Ảnh luôn bị báo mờ**: vệ sinh lens/tăng sáng hoặc giảm thận trọng `BLUR_THRESHOLD` (mặc định `12`; UI hiển thị điểm đo thực tế).
- **Nhận sai người**: tăng `FACE_THRESHOLD`, đăng ký lại với ảnh rõ/nhiều góc và kiểm thử trên dữ liệu thật.
- **Google 403**: bật cả Sheets API và Drive API, rồi share spreadsheet đúng email Service Account.
- **Google credentials not found**: dùng đường dẫn tuyệt đối với dấu `/` hoặc đặt giá trị path hợp lệ trong `.env`.
- **Mất Internet**: bản ghi giữ `PENDING` hoặc `ERROR`; không cần can thiệp, worker/màn hình sync sẽ thử lại.
- **Employee chưa có face**: đăng ký tại Register Face; recognizer chỉ load các employee có embedding.
- **Database chưa tồn tại**: đây là trạng thái bình thường; schema tự tạo. Nếu báo permission, chuyển project sang thư mục user có quyền ghi.

## Cấu trúc project

```text
app.py                       Streamlit entry point
config.py                    .env configuration
database/                    schema, connection, CRUD
face/                        detection, embedding, matching, registration, liveness
attendance/                  rules and transactional service
camera/                      reconnecting capture and realtime worker
google_sheets/               API adapter and retry worker
ui/                          eight Streamlit pages
utils/                       logging and helpers
scripts/create_demo_data.py  optional demo seed
tests/                       unit tests
data/                        runtime database/face data
logs/                        rotating application log
```

## Ghi chú bảo mật và vận hành

- Không đưa `.env`, RTSP URL có mật khẩu hoặc Service Account JSON lên Git.
- SQLite dùng WAL và busy timeout để UI, camera worker và sync worker truy cập an toàn hơn.
- Anti-spoofing đang mặc định tắt. Không nên triển khai tại môi trường kiểm soát truy cập bảo mật cao trước khi tích hợp và đánh giá một model liveness phù hợp.
- Thời gian hiện dùng timezone local của Windows; hãy đặt timezone máy đúng trước khi vận hành.
# Tài khoản cá nhân (bản Spatial Analytics)

- Nhân viên đăng nhập bằng mã NV; mật khẩu ban đầu `123`, bắt buộc đổi lần đầu (ít nhất 8 ký tự).
- Quản lý giữ mật khẩu hiện tại. Trang Đăng ký có tab Nhân viên và Đăng ký khuôn mặt.
- Nhân viên chỉ đăng ký khuôn mặt lần đầu và xem/sửa lịch của bản thân. Đăng ký lại khuôn mặt do quản lý thực hiện qua tải ảnh và xác nhận ghi đè.
- Quản lý đặt lại mật khẩu tại Đăng ký → Nhân viên → Tài khoản nhân viên. Reset vô hiệu hóa phiên cũ.
- Năm lần đăng nhập sai khóa tài khoản 5 phút. Các hồ sơ hiện có được tạo tài khoản tự động, không thay đổi attendance hay camera.
