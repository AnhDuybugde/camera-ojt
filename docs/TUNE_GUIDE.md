# Tune Guide — những phần cần tune thủ công sau đợt dọn logic

Ngày dọn: 2026-09-15. Cập nhật better-version: 2026-09-16 (recognition/
attendance/status lên bản dùng thực tế). Các giá trị dưới là **điểm khởi đầu
hợp lý**, không phải tối ưu cuối. Tune theo thứ tự ưu tiên, mỗi lần 1 nhóm.

## 0. Việc đã làm sẵn (không cần tune lại)

### Đợt 1 — dọn logic + lightweight
- Xóa `HistogramEmbedding` (`workstate/reid.py`): production cấm fallback histogram.
- Xóa `identity.reid_backend` (config chết, không ai đọc).
- Xóa `name_map/employee_map` hardcode 4 người trong `default.yaml` → `registry.json`
  là nguồn thật duy nhất.
- OSNet mặc định `osnet_x0_25` + FP16 CUDA + `cudnn.benchmark` (`workstate/reid.py`).
- InsightFace `det_size` mặc định `320` (nhanh ~3-4x so với `640`).
- MediaPipe Hands: `static_image_mode=False`, `max_num_hands=1` (video mode, nhẹ CPU).
- `detection.confidence_threshold` `0.10 → 0.25` (bớt ghost, ByteTrack low-conf vẫn nối).
- `attendance.debounce_hits` `1 → 2` (chống tick nhầm 1 frame mặt lạ).
- `workstate.move_ratio` `0.15 → 0.0` = presence-only cho tới khi ROI đo thật.
- `voice.bridge_port` `8765 → 8767` (8765=stream, 8766=supervisor).
- `voice.unknown_phrase` troll → `"Xin chào quý khách"`.

### Đợt 2 — recognition/attendance/status better-version (2026-09-16)
- Recognition: enroll đa ảnh `data/images/<person_id>/*.jpg` (giữ tương thích 1 file
  legacy), embedding chính = mặt nét nhất + seed prototypes đa góc; lọc
  `min_face_score=0.5`; `min_person_area_px` `2000 → 8000`; prototype chỉ grow khi
  `quality>=0.25 + sharpness>=40`; `save_prototype` chống ghi đè.
- Attendance: preload tick trong ngày từ Supabase sau restart (hết tick trùng);
  `prune()` dọn window/best-shot GID chết mỗi flush; `needs_review=true` khi
  `face_score<0.80` (tick biên cần admin duyệt).
- Status: channel B (cửa) ép presence-only (không dùng ROI của phòng A);
  `--away-grace-s` `1.5 → 5.0`, `--out-after-s` `20 → 60` (hết flip do RTSP flicker);
  `inroom_min_interval_s` `3600 → 600` (dashboard hết kẹt 1h); thêm
  `absent_fallback_s=900` (vắng A quá lâu không có B vẫn kết Out, 0=tắt).

## 1. Detection (YOLO) — tune đầu tiên, ảnh hưởng mọi thứ sau

File: `config/default.yaml` → `detection`, `camera`.

| Param | Hiện tại | Khi nào tăng/giảm | Cách kiểm |
|---|---|---|---|
| `confidence_threshold` | 0.25 | Đông người + nhiều ghost (ID nhảy) → tăng 0.30-0.35. Người xa hay miss → giảm 0.15-0.20 | Đếm ghost trong `output/ghosts/` sau 10 phút |
| `nms_iou_threshold` | 0.50 | Người đứng sát nhau bị gộp box → giảm 0.40-0.45. Box vỡ vụn → tăng 0.55-0.60 | Xem overlay cam A/B |
| `image_size` | 640 | GPU yếu/lag → 512. Người xa >5m miss nhiều → 768-800 (nặng) | `benchmark_inference.py --frames 30` |
| `nested_box_containment_threshold` | 0.85 | Vẫn còn box con trong box lớn → giảm 0.75-0.80. Mất người đứng sau → tăng 0.90 | Test video 2 người chồng lấn |
| `camera.process_every_n_frames` | 2 | CPU/GPU đuối → 3 (nhưng tracking rời rạc hơn). Cần mượt → 1 | `StageMetrics` log mỗi 60s |
| `--min-area` (CLI) | 2000 | Phòng nhỏ/camera xa, người bé bị lọc → giảm 1200-1500. Nhiều nhiễu xa → tăng 2500-3500 | Xem log ghost `<10 hits` |

Chú ý: `yolo26s.engine` build cho sm_89, TRT 10.9. Đổi GPU/model phải rebuild:
`yolo export model=yolo26s.pt format=engine half=True imgsz=640 device=0 dynamic=True batch=16`.

## 2. Tracking + Global Identity — tune thứ hai

File: `tracking`, `identity`.

| Param | Hiện tại | Gợi ý tune |
|---|---|---|
| `tracking.track_high/low/new` | 0.50/0.10/0.60 | Giữ nguyên trừ khi ID đứt nhiều khi che: thử `high 0.40, new 0.50`. Đừng để `low > high`, `new < high` (config validator sẽ báo) |
| `tracking.track_buffer` | 90 (~7s) | Hành lang che lâu → 120. Phòng nhỏ, người ra/vào nhanh → 60 để ID cũ chết sớm |
| `tracking.byte_match_threshold` | 0.80 | ID nhảy khi đông → tăng 0.85. ID đứt khi di chuyển nhanh → giảm 0.70-0.75 |
| `identity.match_threshold` | 0.70 | Tune bằng `calibrate_identity_thresholds.py` trên video có label. Đồng phục giống nhau → tăng 0.75. Người ít, cần nối dai → giảm 0.65 (rủi ro nối nhầm) |
| `identity.min_appearance_similarity` / `named_appearance_floor` | 0.70/0.70 | Cặp song sinh/đồng phục → tăng 0.75. Ánh sáng xấu quá → giảm 0.65 nhưng phải kèm replay eval = 0 false |
| `identity.max_center_distance_ratio` | 0.35 | Camera rung/PTZ → tăng 0.45. Người đông đứng sát → giảm 0.25-0.30 |
| `identity.same_camera_reconnect_*` | 0.08 / 2.0s | Mất 2-3s hay tạo GID mới → tăng distance 0.12 + time 3-4s (đánh đổi ghost) |
| `identity.active_duplicate_similarity` | 0.90 | 2 cam trùng view mà 1 người thành 2 GID lâu → giảm 0.85. 2 người khác nhau bị gộp → tăng 0.93-0.95 |
| `identity.gallery_size` | 8 | Đông người + RAM GPU dư → 10-12. Lag ReID → giảm 5-6 |
| `identity.gallery_refresh_steps` | 5 | Lag → tăng 8-10. ID chập chờn → giảm 2-3 |
| `identity.tentative_min_hits` | 5 | Muốn GID lên nhanh → 3 (nhiều ghost hơn). Nhiều ghost → tăng 7 |
| `identity.temp/long_lost_s` | 10 / 120 | Ca làm việc dài, đi vệ sinh 5-10p vẫn muốn giữ ID → tăng `long_lost 300-600`. Quán đông, ID cũ ám → giảm |
| `identity.reid_model` | `osnet_x0_25` | Muốn chính xác hơn, GPU khỏe → `osnet_x0_5` → `osnet_x1_0`. Nhớ đo lại latency |
| `identity.reid_device` | auto | Ép `cuda` để fail-fast nếu mất GPU, hoặc `cpu` để test không GPU |

Bắt buộc sau mỗi lần đổi: chạy replay eval, target **0 false-employee + 0 uniqueness violation**
(`scripts/evaluate_replay.py`, exit code 2 = fail).

## 3. Face Recognition — tune thứ ba

File: `face`.

| Param | Hiện tại | Gợi ý |
|---|---|---|
| `match_threshold` / `min_margin` | 0.70 / 0.10 | Mặt góc nghiêng hay Unknown → đừng vội giảm threshold; enroll thêm 3-5 góc trước. Chỉ giảm tới 0.60-0.65 nếu đã enroll đủ mà vẫn miss, và phải kèm replay |
| `consensus_hits` / `window_s` | 3 / 3.0s | Check-in chậm → giảm hits 2. Mặt lạ hay tick nhầm → tăng hits 4-5 |
| `det_size` | 320 | Mặt xa >4m miss → tăng 480-640 (nặng CPU/GPU). Kiosk gần 1-2m giữ 320 |
| `min_face_px` / `min_blur_variance` | 40 / 40.0 | Phòng tối hay Unknown → giảm blur 25-30 (đánh đổi ảnh mờ). Mặt xa nhỏ → giảm face_px 30-32 |
| `min_person_area_px` | 2000 | Đồng bộ với `--min-area` CLI |
| `process_every_k` | 3 | Check-in trễ >2s → giảm 2. CPU đuối → tăng 4-5 |
| `gallery_accept_threshold` / `max_prototypes` | 0.80 / 10 | Prototype bẩn (nhận nhầm sau 1 thời gian) → tăng accept 0.85 + giảm max 5. Người thay đổi ngoại hình (kính/khẩu trang) → giảm accept 0.75 |
| `face_device` | auto | Có `onnxruntime-gpu` + CUDA Toolkit thì để auto (lên CUDA). Không thì ép `cpu` cho ổn định |

Việc tay (không phải param): enroll mỗi người 3-5 ảnh (chính diện, trái/phải 30°, cười/không cười,
có/không kính), đủ sáng, 1 mặt/ảnh.

## 4. Attendance — kiểm tra logic, ít param

File: `attendance`.

- `debounce_hits=2, window_s=8.0`: nếu vẫn tick nhầm người lạ → tăng hits 3. Nếu check-in chậm
  (đứng 3s chưa tick) → kiểm tra `face.consensus` trước, đừng giảm debounce về 1.
- `active_hour_start/end=null` = tick cả đêm. Phòng lab 24/7 thì giữ. Văn phòng hành chính
  nên set `6 → 22` để đêm bảo vệ/test không tick.
- Còn thiếu: preload attendance từ Supabase khi restart (hiện chỉ nhớ trong RAM + queue).
  Nếu restart giữa ngày hay tick trùng, ưu tiên fix code này trước khi tune số.

## 5. Status / Workstate — phụ thuộc đo đạc thực tế, không tune số mù

1. Đo `analytics.calibration` (4 điểm sàn thật) + `workstations` (core/extended mét sàn).
   Hiện `workstations: []` + `move_ratio: 0.0` = presence-only (thấy = Working) — đúng và an toàn.
2. Chỉ bật lại `move_ratio 0.10-0.15` nếu camera chưa cố định và chấp nhận false Away.
   Khi đã có ROI thật, giữ `move_ratio=0.0`.
3. `grace_s=1.5, dwell_s=2.0, hysteresis 0.3`: phòng nhiễu biên (đi qua lại mép bàn hay flip)
   → tăng grace 2.5 + dwell 3.0. Muốn nhạy (ra khỏi ghế báo ngay) → giảm grace 1.0 + dwell 1.0.
4. `away_grace_s=1.5, out_after_s=20.0` (CLI): người đi vệ sinh 5p hay bị `AWAY` →
   tăng `out_after 45-60`. Muốn báo rời phòng nhanh → giảm 10-15.
5. `room_fusion.leave_confirm_window_s=300, inroom_min_interval_s=3600`: đừng giảm
   `inroom_min` dưới 600 nếu không muốn spam Supabase.

## 6. Voice / Wave — tune cuối cùng, sau khi face ổn

File: `voice`. Port hiện tại: stream `8765`, supervisor `8766`, voice bridge `8767`.

- `wave_min_reversals=4, window 1.5s`: vẫy không ăn → giảm reversals 3. Tay đi bộ hay trigger
  nhầm → tăng 5-6 + tăng `wave_cooldown 30 → 60`.
- `wave_every_k=6, max_people=2`: CPU cao do MediaPipe → tăng every_k 8-10, giảm max_people 1.
  MediaPipe vẫn chạy CPU (không có GPU wheel ổn định trên Windows) — đây là bottleneck
  đã biết, đừng ép video 25fps cho nhánh này.
- `cooldown_s=60, command_ttl_s=5.0`: phòng ồn/khách đông bị chào dồn → tăng cooldown 120.
  Loa camera ngắt câu → tăng `talk_tail 0.3 → 0.5`.
- `unknown_phrase="Xin chào quý khách"`: đổi câu theo văn hóa công ty trong yaml, không sửa code.
- Edge-TTS cần mạng lần đầu, sau cache `output/voice_cache`. Mất mạng hoàn toàn →
  prewarm trước (`greeter.prewarm`) rồi mới bật `--greet`.

## 7. Thứ tự tune khuyến nghị (checklist)

- [ ] 1. Cố định camera, đo calibration + workstation ROI thật.
- [ ] 2. Tune YOLO `conf` + `min-area` tới khi ghost `<5%` trong 10 phút.
- [ ] 3. Tune `identity.match_threshold` bằng calibration video, replay eval = 0 false.
- [ ] 4. Enroll lại face 3-5 ảnh/người, kiểm tra check-in <2s.
- [ ] 5. Set `active_hour` + kiểm tra preload attendance sau restart.
- [ ] 6. Bật ROI workstate, tắt hẳn interim (`move_ratio=0`).
- [ ] 7. Cuối cùng mới bật `--greet` và tune wave khi camera online (`test_imou_voice.py`).

## 8. Code còn giữ lại nhưng không dùng production (đừng xóa vội)

- `tracking/iou.py` (`IoUTracker`) + `tracking/identity.py` (`PersistentIdentityTracker`):
  chỉ dùng cho `scripts/run_pipeline.py` demo 1 cam + smoke test. Production dùng ByteTrack.
- `tracking.max_lost_frames` / `iou_threshold`: chỉ cho demo trên, ByteTrack không đọc.
- `face/reid.py` (`FaceReIDEmbedding`): ReID bằng mặt, hiện chưa cắm vào pipeline
  (pipeline dùng OSNet body). Giữ cho experiment, không dùng production.
- `voice` backend `local` (ffplay): debug không loa camera. Production dùng `imou_web`.
