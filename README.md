# Camera OJT — Unified Camera Platform

Điểm danh có hướng qua cửa, theo dõi hiện diện và trợ lý tiếng Việt. Imou,
webcam và video replay dùng chung pipeline. Giao diện chính là Streamlit;
backend sở hữu nhận diện, nghiệp vụ, đồng bộ và đăng ký khuôn mặt.

## Chạy

Python 3.10+, FFmpeg. Cài PyTorch đúng thiết bị trước khi cài project.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U 'setuptools>=69,<80' wheel
# Máy CPU:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints.txt -e '.[ui,face,reid,voice,dev]'
```

Model và dữ liệu cá nhân không nằm trên GitHub. Có thể dùng wheel InsightFace
trong `apps/attendance/vendor/` nếu môi trường không build được extension gốc.
Giữ `.env` thật ở gốc repo; dùng `.env.example` làm danh sách cấu hình.

```bash
camera-ojt doctor
# Chạy backend + giao diện + hai luồng Imou:
camera-ojt run
# Máy chưa dùng được GPU:
camera-ojt run --device cpu
# Webcam (một kênh):
camera-ojt run --profile webcam --device cpu
# Chỉ quản trị dữ liệu, không mở camera/mic:
camera-ojt run --no-camera
```

Giao diện: **http://127.0.0.1:8501**. API nghiệp vụ: `127.0.0.1:8767`.
Luồng hình: `127.0.0.1:8765`. Ctrl+C dừng các tiến trình con.
`camera-ojt run` tự tạo token nội bộ cho các tiến trình, không đưa token đó xuống
trình duyệt; ảnh trực tiếp dùng URL ký có thời hạn.

Mật khẩu hiện có được giữ nguyên. Khi tạo mới admin, dùng `CAMERA_ADMIN_PASSWORD`
hoặc lấy mật khẩu khởi tạo ở `apps/attendance/data/bootstrap-accounts.json`.
Tài khoản nhân viên mới có mật khẩu tạm riêng trong `employee-invitations.json`
cùng thư mục; quản trị có thể reset trên UI. Các file này có quyền đọc hạn chế
và không được commit. Nhân viên phải đổi mật khẩu trước khi sử dụng.

## Dữ liệu nhận diện và đồng bộ

```bash
# Sao lưu SQLite nhất quán, rồi áp dụng migration local bổ sung:
python tools/migrate_local.py
# Kiểm kê bộ ảnh gán nhãn (không thay đổi dữ liệu):
python tools/import_enrollment.py
# Nhập toàn bộ mẫu hợp lệ, giữ ảnh và metadata nguồn:
python tools/import_enrollment.py --apply --device cpu
# Migration cloud, cần psycopg2/psycopg2-binary và CONNECT_STRING:
python tools/migrate_cloud.py --apply
python tools/sync_cloud.py
```

Gallery dùng chung `employee_id`, model/version và checksum ảnh. Các vector
khác model không được trộn. Mẫu vận hành không tự động sửa gallery mặc định.
Các mẫu đăng ký mới trên UI được xử lý ở backend và cập nhật gallery runtime.

Sự kiện ra/vào cần quan sát chuỗi **trong phòng → cửa → ngoài phòng**, hoặc
chiều ngược lại. Mặc định dùng camera B và vùng R2/R3 hiện có. Với webcam hoặc
góc camera khác, cần cấu hình vùng đúng trước khi dùng điểm danh thật. Mất dấu
không tự tạo check-out. Danh tính không chắc được đưa vào trang **Hệ thống AI &
Trợ lý → Sự kiện cần xác nhận danh tính**.

SQLite giữ sự kiện và outbox trước khi gửi. Supabase giữ bản quản trị tập trung,
đồng bộ hồ sơ, lịch, điểm danh, audit và embedding có phiên bản. Sửa đồng thời
tạo xung đột trên trang quản trị; quản trị chọn bản cloud hoặc local. Mất điện
khôi phục sau restart; muốn tiếp tục ghi hình khi mất điện cần nguồn dự phòng.

Gemini xử lý câu hỏi, Tavily tìm kiếm ngoài. Cần `GEMINI_API_KEY`,
`GEMINI_MODEL`, `TAVILY_API_KEY` để chạy các chức năng tương ứng. Câu hỏi đơn
giản hỗ trợ trả lời local. Lỗi nhà cung cấp không dừng điểm danh.

## Cấu trúc

```text
apps/attendance/       Streamlit và nghiệp vụ đang chuyển tiếp
src/camera_tracking/   API, runtime chung, vision, voice và lưu trữ
backend/               Facade docs: backend/app/* ↔ src/camera_tracking/*, apps/attendance/*
frontend/              Facade docs: frontend/src/* ↔ apps/attendance/ui|integration/*
config/               Cấu hình camera/model/vùng, không chứa bí mật
migrations/supabase/   Migration cloud có phiên bản
tools/                Import, migration, đồng bộ, kiểm tra publish
scripts/              Entry point tương thích và công cụ camera
tests/                Kiểm thử nghiệp vụ và nhận diện
var/                  Embedding, backup, báo cáo local — không commit
```

`ai-mind-attendance` là đường dẫn tương thích tới `apps/attendance`.
Xem [bản đồ cấu trúc chi tiết](docs/STRUCTURE.md),
[camera inventory](config/cameras.yaml) và [docker-compose](docker-compose.yml).
Dashboard React đã được thay thế. Các script `run_workstate*.py` vẫn hoạt động,
nhưng lệnh `camera-ojt run` quản lý đầy đủ backend và UI.

## Kiểm chứng và giới hạn

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_unified_platform.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_backend_connection.py
python tools/smoke_ui.py
python tools/check_backend_connection.py  # backend :8767 + pipeline :8765 healthcheck
camera-ojt doctor  # modules + keys + ports + backend reachability
camera-ojt replay --source-a path/to/video-a.mp4 --source-b path/to/video-b.mp4 \
  --device cpu --identity-log var/predictions.csv
python scripts/evaluate_replay.py --truth annotations.csv --predictions var/predictions.csv
```

Zero false assignments trên holdout, tỷ lệ nhận đúng và độ trễ là **cổng nghiệm
thu cần đo**, không phải kết quả đã được chứng minh trên mọi tình huống thực tế.
Xem [trạng thái triển khai](docs/IMPLEMENTATION.md),
[kiến trúc](docs/UNIFIED_PLATFORM.md) và [backlog](docs/ROADMAP.md).
