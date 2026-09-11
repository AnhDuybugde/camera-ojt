# Chiến lược Workstate xuyên camera (MVP)

## 1. Định nghĩa bài toán
- `WORKING`: detect trong ROI bàn (channel A).
- `AWAY_SHORT`: rời ROI > `leave_grace_s` (mặc định 10s), lưu embedding lúc rời.
- `RESTROOM`: khớp ở hallway B trong `corridor_match_window_s` (300s).
- `OUT_OF_OFFICE`: timeout không thấy ở B, hoặc khớp ở vùng `exit_door`, hoặc `RESTROOM` quá `restroom_return_window_s` (900s).
- `RETURNED`: quay lại ghế (thoáng qua ~3s rồi về `WORKING`).

## 2. Vấn đề kỹ thuật cốt lõi
- **Re-ID xuyên camera**: MVP dùng `HistogramEmbedding` (HSV thân dưới, cosine).
  Interface `EmbeddingExtractor` giữ nguyên để thay bằng ArcFace/InsightFace/OSNet sau.
- **Time-window**: cấu hình trong `config/default.yaml` -> `workstate`.
- **ROI**: pixel polygon theo từng camera. Channel A: `seats[].polygon`.
  Channel B: `corridor.hallway` + `corridor.exit_door`. Phải chỉnh lại cho khớp góc cam thật.

## 3. Kiến trúc
```
Cam A ─┐
       ├─ YOLOv11 detect -> IoUTracker -> ROI check + embedding -> WorkStateEngine -┐
Cam B ─┘                                                                             ├─> console + JSONL
       YOLO detect -> IoUTracker -> embedding + zone (hallway/exit) ────────────────┘
```

## 4. Chạy MVP
```bash
python scripts/run_workstate.py --config config/default.yaml --display
python scripts/run_workstate.py --source-a 0 --source-b data/samples/hallway.mp4 --display
# RTSP IMOU tự lấy từ .env (IMOU_IP/USER/PASSWORD), override bằng --channel-a/--channel-b
```
Event log: `output/workstate_events.jsonl`. Nhấn `q`/ESC thoát.

## 5. Thực tế cần lường trước
- 2 người rời gần nhau -> dễ nhầm histogram. Giảm bằng ngưỡng `similarity_threshold` cao (0.6-0.7) + face khi có.
- Ánh sáng A/B khác nhau -> histogram lệch. Chuẩn hóa HSV giúp một phần.
- Camera B sau lưng -> fallback trang phục là đúng hướng, nhưng cần OSNet về sau.
- **Riêng tư**: đây là giám sát nhân viên sâu. Cần thông báo minh bạch + tuân thủ nội quy/pháp luật.

## 6. Hướng nâng cấp
1. Thay IoUTracker bằng ByteTrack/DeepSORT khi đông người.
2. Thêm face embedding (InsightFace) + fusion với body embedding.
3. Lưu DB (SQLite/Postgres) + dashboard real-time.
4. Hiệu chuẩn ROI bằng tool click polygon thay vì sửa YAML tay.
