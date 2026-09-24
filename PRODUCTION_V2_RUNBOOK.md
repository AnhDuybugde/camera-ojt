# Camera OJT — Production V2 Runbook

## Mục tiêu

Bản này tách hệ thống thành hai đường an toàn:

1. **Perception/Identity** luôn chạy để tracking, face recognition, dashboard và voice có dữ liệu.
2. **Attendance side effects** chỉ được ghi khi policy cho phép. Ở `APP_ENV=production`, auto attendance fail-closed nếu chưa có liveness provider.

Bé Xinh có hai backend:

- `live` (chọn bằng cấu hình): Gemini Live native audio, giữ ngữ cảnh nhiều lượt và phát PCM trực tiếp ra loa camera theo chunk.
- `classic`: wake STT -> Gemini 3.5 Flash -> ZeroTTS, dùng làm fallback.

## 1. Cài dependencies

```bash
cd backend
python -m pip install -e ".[face-gpu,store,reid,voice,dev]"
python -m pip install -U google-genai
```

Máy production NVIDIA phải thấy `CUDAExecutionProvider` và `torch.cuda.is_available() == True`.

## 2. Database

Chạy `backend/supabase/schema.sql` cho hệ mới, hoặc migration:

```text
backend/supabase/migrations/20260924_production_hardening.sql
backend/supabase/migrations/20260924_attendance_lifecycle_v2.sql
```

Attendance hiện có các field chính:

- `check_in_at`
- `check_out_at`
- `status`: `PRESENT | TEMP_OUT | CHECKED_OUT`
- `face_score`
- `liveness_score`
- `verification_method`
- `automatic`
- `needs_review`

`attendance_audit_log` dành cho lịch sử điều chỉnh nghiệp vụ/HR.

## 3. Shadow mode — bắt buộc trước pilot

```env
APP_ENV=production
ATTENDANCE_AUTOMATION_ENABLED=false
```

Trong mode này:

- YOLO/ByteTrack/ReID vẫn chạy.
- Face recognition vẫn chạy.
- Bé Xinh vẫn hỏi được ai đang có mặt.
- Không có biometric attendance write tự động.

Đây là mode dùng để đo ID switch, unknown rate, face FAR/FRR và so sánh với hệ chấm công hiện tại.

## 4. Liveness

Production auto attendance hỗ trợ provider contract qua HTTP:

```env
BIOMETRIC_LIVENESS_PROVIDER=http
BIOMETRIC_LIVENESS_URL=http://127.0.0.1:8890/verify
BIOMETRIC_LIVENESS_TOKEN=...
BIOMETRIC_LIVENESS_THRESHOLD=0.80
BIOMETRIC_LIVENESS_TIMEOUT_SECONDS=1.5
```

Request:

```json
{"image_base64":"...jpeg..."}
```

Response tối thiểu:

```json
{"live":true,"score":0.97,"reason":"verified"}
```

Timeout/error/score thấp đều **fail closed** và không tạo attendance.

Chỉ sau khi provider đã audit và pilot đạt KPI mới bật:

```env
ATTENDANCE_AUTOMATION_ENABLED=true
ATTENDANCE_ENTRY_CHANNEL=B
```

## 5. Attendance lifecycle

Lifecycle chuẩn:

```text
ABSENT
  -> CHECK_IN -> PRESENT
PRESENT
  -> TEMP_OUT -> TEMP_OUT
TEMP_OUT
  -> RETURN -> PRESENT
  -> CHECK_OUT -> CHECKED_OUT
```

Face recognition là identity evidence; room/door transition là business event. `AttendanceLifecycleEngine` nằm tại:

```text
backend/src/camera_tracking/attendance/lifecycle.py
```

## 6. Bé Xinh realtime

Production `.env`:

```env
GEMINI_API_KEY=...
BE_XINH_VOICE_ENABLED=true
BE_XINH_REALTIME_MODE=live
GEMINI_LIVE_MODEL=gemini-3.8-live
BE_XINH_CONVERSATION_IDLE_SECONDS=45
```

Luồng:

```text
local wake "Bé Xinh ơi"
   -> Gemini Live session
   -> raw PCM 16k input
   -> native PCM 24k response
   -> IMOU VisualTalk streaming speaker
```

Sau khi wake một lần, người dùng có thể hỏi tiếp trong cùng conversation window mà không cần gọi lại tên. Live session giữ context và có function calling tới các tool camera/runtime hiện có.

### Fallback

Nếu Live API không phù hợp với hạ tầng mạng:

```env
BE_XINH_REALTIME_MODE=classic
GEMINI_MODEL=gemini-3.5-flash
```

Classic mode giờ cũng mang theo vài lượt hội thoại gần nhất, nhưng latency sẽ cao hơn vì có STT + text generation + TTS.

### Full duplex / ngắt lời

Mặc định hệ thống giữ half-duplex vì mic và loa cùng nằm trên camera IMOU; đây là cách chống Bé Xinh nghe lại chính giọng của mình.

`--full-duplex` chỉ nên bật sau khi đã chứng minh camera/đường audio có AEC (acoustic echo cancellation) đủ tốt. Không có AEC, true barge-in dễ tạo self-trigger/loop.

## 7. Start

Dashboard `SystemController` mặc định launch realtime assistant khi:

```env
BE_XINH_REALTIME_MODE=live
```

Có thể chạy tay:

```bash
cd backend
python scripts/be_xinh_live_assistant.py --model gemini-3.8-live
```

Classic fallback:

```bash
python scripts/be_xinh_assistant.py --model gemini-3.5-flash
```

## 8. Preflight

```bash
cd backend
PYTHONPATH=.:src python scripts/preflight_production.py --strict
```

Strict release phải không có blocker.

## 9. Regression

```bash
cd backend
PYTHONPATH=.:src pytest -q
```

Tại thời điểm đóng gói V2: **332 backend tests passed**; toàn bộ frontend tests và dashboard production build đều pass trong môi trường review.

## 10. Rollout đề xuất

1. Shadow mode 3–7 ngày.
2. Pilot 5–20 nhân viên, vẫn giữ phương thức chấm công hiện tại làm ground truth.
3. Đo false accept, false reject, unknown rate, ID switch, missed entry/exit và voice first-audio latency.
4. Calibrate face threshold trên đúng camera/góc sáng thực tế.
5. Xác nhận liveness provider, retention, consent/purpose, audit và manual correction với HR/pháp lý.
6. Bật auto attendance theo nhóm nhỏ.
7. Chỉ rollout toàn công ty sau khi sai lệch bảng công đạt ngưỡng nội bộ đã phê duyệt.
