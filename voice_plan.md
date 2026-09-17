# Voice Trigger trong phòng ồn — Plan thực thi

Ngày chốt: 2026-09-17. Trạng thái: CHƯA LÀM (tài liệu để thực thi sau).

## Định hướng đã chốt
- **Ưu tiên: thà miss còn hơn nhầm.** Mọi ngưỡng nghiêng về phía chắc mới chào;
  chấp nhận miss ~25% ở vị trí gọi chuẩn (bù bằng kênh vẫy tay / face-trigger vẫn chạy).
- **Phòng có cả 2 loại ồn:** ồn đều (điều hòa/quạt/máy) + ồn lộn xộn (nhiều người nói, TV, nhạc).
- **Được thêm thư viện.**

## Giá trị hiệu lực trong code (yaml thắng default code)
| Tham số | Giá trị hiệu lực (`config/default.yaml`) | Vị trí |
|---|---|---|
| `voice_vad_threshold` | 0.012 | `scripts/run_workstate.py` ~dòng 1684 |
| `voice_min_seg_rms` | 0.008 | `scripts/run_workstate.py` ~dòng 1685 |
| `voice_stt_model` / lang | small / vi | `scripts/run_workstate.py` ~dòng 1682–1683 |
| `voice_max_no_speech_prob` / `voice_min_avg_logprob` | 0.55 / −0.85 | `scripts/run_workstate.py` ~dòng 1686–1689 |
| `voice_trigger_window_s` / `voice_trigger_inhibit_s` | 5.0 / 12.0 | `scripts/run_workstate.py` ~dòng 1674–1675 |
| Filler cho phép | a/da/oi/camera/cam/hey/hi | `src/camera_tracking/voice/voice_trigger.py` |
| Luật từ khóa | ≤ 6 token, keyword + filler | `is_voice_trigger()` cùng file trên |
| Lọc audio ffmpeg | `highpass=f=80,afftdn=nr=12:nf=-25` | `src/camera_tracking/voice/rtsp_voice_listener.py` |
| Cắt đoạn VAD | mở sau 90ms vượt ngưỡng, tối thiểu 300ms, im 450ms cắt, tối đa 3s | cùng file trên |

## Các bước thực hiện (đúng thứ tự)

### P0 — Đo nền ồn thật (script calibrate, không đoán)
- Viết `scripts/calibrate_mic.py`: mở mic RTSP 30 giây giờ ồn điển hình.
- In ra: RMS nền trung bình/đỉnh; RMS của 3 tiếng "xin chào" gọi thử ở vị trí người đứng.
- Output: ngưỡng VAD khuyến nghị (= đỉnh nền × 2, kẹp 0.02–0.08) + biên SNR khả thi.
- Chạy lại mỗi khi đổi mùa / máy lạnh / bố trí phòng.

### P1 — Ngưỡng động bám nền ồn (trị ồn đều)
- Thay ngưỡng VAD tĩnh bằng: theo dõi nền ồn qua trung bình trượt các frame yên;
  ngưỡng = max(nền × 3, sàn 0.02).
- Tham số tune: hệ số nhân nền (khởi điểm 3), tốc độ bám, sàn tuyệt đối.

### P2 — Silero VAD thay "tai" energy (trị ồn lộn xộn, bước nặng nhất)
- Giữ energy-RMS làm vòng ngoài rẻ tiền; thêm Silero VAD (CPU ok) chấm từng frame
  30ms có phải giọng người không. Chỉ frame nào cả hai cùng gật mới tính là tiếng nói.
- Tham số tune: ngưỡng tin cậy Silero (khởi điểm 0.6, nghiêng cao), số frame liên
  tiếp để mở/đóng đoạn.
- Làm sau cùng trong nhóm code; nếu P1+P5 đã đạt P7 thì có thể bỏ qua.

### P3 — Cửa SNR ở cấp đoạn
- Đoạn cắt ra chỉ đem đi STT khi to hơn nền đang theo dõi ≥ ~10 dB.
- Loại tại chỗ: ồn to đều + tiếng nói xa nhỏ → khỏi tốn lượt STT.

### P4 — STT nhỏ lại + siết cửa chất lượng
- medium → small (thử tiếp base nếu P7 vẫn đạt). Model càng nhỏ càng ít bịa chữ
  từ tiếng ồn, càng nhẹ CPU.
- Siết `max_no_speech_prob` ~0.4, `min_avg_logprob` ~−0.7 (chốt số theo log thật).
- Giữ beam=1.

### P5 — Siết luật từ khóa + gắn chặt với mặt
- Từ khóa đứng gần như một mình: tối đa 1 tiếng đệm; bỏ `camera/cam` khỏi filler.
- Voice chỉ hiệu lực khi trong ~3 giây gần đó có mặt tươi **và mặt đủ lớn**
  (loại tiếng gọi từ xa ngoài khung hình — mic toàn cục nên đây là cửa chống nhầm
  quan trọng nhất).
- Giữ nguyên: 1 câu hello greet đúng 1 người gần nhất, inhibit sau phát, window 5s.

### P6 — Inhibit theo đuôi vang thật
- Đo trong phòng: inhibit = câu chào dài nhất + ~2 giây dội (dự kiến 6–8s thay 4s
  hiện tại) để đuôi vang loa không thành tiếng gọi mới.

### P7 — Nghiệm thu trong phòng ồn thật (đạt cả 3 mới xong)
- (a) Ồn 5 phút không ai gọi → **0 trigger nhầm**, lượt STT vô ích giảm ≥ 80%.
- (b) Gọi "xin chào" 20 lần ở vị trí người đứng → ăn ≥ 15.
- (c) 10 câu chuyện thường có lẫn từ "hello" → **0 trigger nhầm**.

## Thứ tự + rủi ro
P0 → P1 → P5 (nhẹ, hiệu quả ngay) → P4 → P6 → P2 (nặng nhất, chỉ làm khi cần) → P7.
Mỗi bước xong đo lại bằng kịch bản P7 rồi mới đi tiếp.

## Lệnh chạy kiểm chứng (dùng khi thực thi từng bước)
```bash
# Pipeline chính: camera + chào + voice trigger
python scripts/run_workstate.py --greet --voice-trigger

# Dashboard (terminal khác)
cd dashboard && npm run dev

# Test loa P2P trực tiếp (không cần pipeline)
python scripts/test_imou_p2p_voice.py --greeting unknown
python scripts/test_imou_p2p_voice.py --greeting unknown --channel 2

# Tests sau mỗi bước code
pytest tests/ -x -q
```
