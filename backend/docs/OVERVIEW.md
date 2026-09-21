# Camera-OJT — Tổng quan hệ thống

> Bản đồ cấp cao: luồng hoạt động, model nào làm task gì, số liệu đã kiểm
> chứng từ code/config, và cơ chế chống trùng lặp / spam / nhiễu.
> Chi tiết từng module xem các file còn lại trong `docs/`
> (`IDENTITY_ARCHITECTURE.md`, `P2P_TALK.md`, `TUNE_GUIDE.md`, ...).

Có 2 bản tracking + 3 bản voice (chọn đúng cặp khi chạy):

| Bản tracking | Lệnh | Camera/mic/loa | Dùng khi nào |
|---|---|---|---|
| Chính thức (camera Imou) | `python scripts/run_workstate.py` | 2 cam Imou (RTSP) + mic/loa camera qua RTSP + P2P | Triển khai phòng thật |
| Local (laptop) | `python scripts/run_workstate_local.py` | Webcam + mic/loa laptop | Dev/test không cần camera |

| Bản voice Hà Linh | Mic/loa | Đi với |
|---|---|---|
| `halinh_assistant.py` (chính thức) | mic RTSP + loa P2P camera | Camera Imou |
| `halinh_assistant_local.py` (hậu tố `_local` = test) | mic pulse + loa laptop (paplay/aplay/ffplay) | Webcam |

Lưu ý vận hành:

- `run_workstate[_local].py` **tự kèm voice**: mặc định `--halinh` bật, supervisor
  chạy Hà Linh ở subprocess riêng, crash thì restart tối đa 3 lần
  (`--halinh-restarts`, `--no-halinh` để tắt). Không cần mở voice tay khi
  chạy tracking.
- Chạy voice rời vẫn được (test/debug): `halinh_assistant[_local].py`.
- Quy ước tên: hậu tố `_local` = chạy trên laptop (webcam/mic/loa máy);
  không hậu tố = bản chính thức (camera Imou).

---

## 1. Sơ đồ luồng tổng

```text
                    ┌──────────────────────────────────────┐
                    │  NHÌN (vision, theo frame)           │
  camera/webcam ──▶ │ capture ─▶ YOLO26s ─▶ ByteTrack      │
                    │      ─▶ ReID OSNet (Global ID)       │
                    │      ─▶ Face async (điểm danh)       │
                    │      ─▶ Hands (wave/palm) + workstate│
                    └──────┬───────────────┬───────────────┘
                           │               │ runtime_status.json (2s/lần)
                           ▼               ▼
                    ┌──────────────────────────────────────┐
                    │  NGHE–NÓI (voice, tiến trình riêng)  │
                    │  WAIT_WAKE ─▶ ack ─▶ LISTEN_CMD      │
                    │      ─▶ THINK (Gemini 1 call)        │
                    │      ─▶ SPEAK (TTS ─▶ loa)           │
                    └──────┬───────────────┬───────────────┘
                           │               │ voice_command.json (khi cần)
                           ▼               ▼
                     chào theo gesture ◀──▶ chào/chụp theo lệnh voice
```

Ba file cầu nối duy nhất giữa mắt và miệng (đều trong `output/qa_cache/`):

- **Mắt → miệng**: tracking ghi `runtime_status.json` **mỗi 2s** (ai trong
  phòng, đang làm gì, camera/FPS, điểm danh, sự kiện). Voice đọc để trả lời
  số **thật** + tự nhét 1 dòng ngữ cảnh camera vào mỗi câu hỏi gửi Gemini
  (xem `_RUNTIME_PARTS` trong `voice/qa_tools.py`).
- **Miệng → mắt**: voice ghi `voice_command.json` (`request_greet` /
  `take_snapshot` → `action: greet` / `snapshot`), tracking poll trong main
  loop và thực thi (chào đúng người qua hàng đợi greeter / lưu frame overlay
  vào `output/snapshots/snap_cam_<A|B>_<stamp>.jpg`). Lệnh chỉ nhận khi
  `status == "pending"` và **quá 60s tự đánh `expired`**.
- **Chống hú xuyên process**: bên phát gọi `mark_speaker_busy()` ghi
  `speaker_busy.json`; bên mic (`halinh_assistant_local.py`, greeter)
  đọc `speaker_busy()` và bỏ frame / hủy đoạn đang ghi khi loa bận. File có
  TTL nên crash bên phát không kẹt mic vĩnh viễn.

---

## 2. Nhìn: model nào làm task gì

| Tầng | Model / công nghệ | Task | Số liệu vận hành (từ `config/default.yaml`) |
|---|---|---|---|
| Capture | OpenCV + `ResilientCapture` (RTSP TCP, nobuffer, low-delay) | Lấy frame, tự nối lại khi rớt stream | RTSP latest-frame (chỉ decode frame mới nhất, chống trễ dồn); `process_every_n_frames: 2` trên 25fps |
| Detect | **YOLO26s**, TensorRT FP16 (`yolo26s.engine`), fallback `yolo26s.pt` | Box người mỗi frame | Thuần GPU (~185 frame/s engine vs ~50 `.pt`); `conf 0.25` (recall-first cho ByteTrack low-conf nối lại) + NMS IoU 0.85 + lọc box lồng nhau 0.85 |
| Track ngắn hạn | **ByteTrack** (mỗi kênh độc lập) | Nối box thành tracklet, chịu che khuất | `high 0.50 / low 0.10 / new 0.60`, IoU nối lại 0.3 (dễ nối sau che), `track_buffer 90` ≈ 7s @12fps hiệu dụng; legacy `IoUTracker` chỉ còn trong `run_pipeline.py` demo |
| ID toàn cục | **OSNet x0_25** (ReID body, CUDA FP16) + Hungarian | Giữ Global ID ổn định xuyên tracklet / mất dấu lâu / xuyên camera | Event-driven (chỉ extract embedding khi track mới / mất rồi về / cross-camera); gallery **8 mẫu/ID**; trọng số appearance 0.70 / spatial 0.05 / time 0.15 / channel 0.10; ngưỡng match 0.70 (precision-first: thà bỏ sót còn hơn gán nhầm) |
| Mặt + điểm danh | **InsightFace buffalo_s** (CUDA) | Nhận diện nhân viên, chấm công | Chạy async (`FaceWorker`), once-per-track, không block frame chính; gallery `data/images`, accept 0.70, tối đa 10 prototype/người; 2 track cùng mặt gộp về 1 GID (reconcile) |
| Tay | **MediaPipe Hands** (CPU, gated fps thấp) | Vẫy 5 ngón (wave) / giơ tay (palm) để chào | Wave = palm 5 ngón giữ + lắc đủ đảo chiều; `wave_cooldown_s` / `palm_cooldown_s` = 10s |
| Trạng thái làm việc | Thuật toán vị trí + thời gian (không model) | WORKING / AWAY / RETURNING theo ROI bàn | `away-grace 5s`, `out-after 60s`, `return-stable 2s`; chưa đo ROI thật thì chỉ dùng hiện diện |

---

## 3. Nghe–nói: model nào làm task gì

Máy trạng thái của Hà Linh (giữ nguyên ở cả 3 bản, chỉ khác device mic/loa):

```text
WAIT_WAKE --"Hà Linh ơi"--> ack "Hà Linh nghe nè" --> LISTEN_CMD
  --> THINK (Gemini 1 call JSON --> tool local) --> SPEAK (TTS --> loa)
  --> WAIT_WAKE. Nói gộp "Hà Linh ơi, mấy giờ rồi" thì dùng lệnh luôn.
```

| Tầng | Model / công nghệ | Task | Số liệu thực đo trên máy này |
|---|---|---|---|
| VAD | Energy RMS + `NoiseFloorTracker` (ngưỡng động bám nền) + cổng SNR 10dB | Cắt đoạn có tiếng nói, bỏ ồn đều/quạt | Wake: im 700ms thì cắt, tối đa 4s; lệnh: im 1.4–2s thì cắt, tối đa 10–15s |
| STT wake | **faster-whisper `tiny`** (CPU int8, beam 1) | Chỉ bắt cụm gọi `Hà Linh ơi` | **~0.3s/đoạn**, đúng gần 100% câu ngắn; `medium` cũ mất 3–4s mà hay trả rỗng nên đã bỏ |
| STT lệnh | **faster-whisper `medium`** (CPU int8, beam 1) | Nghe câu lệnh thật | ~3s/đoạn nhưng chuẩn số liệu, tên riêng, câu dài; wake kèm lệnh thì chạy thêm lượt medium để "refine" (log `refine lenh medium`) |
| Hiểu lệnh | **Gemini 3.6-flash, 1 call duy nhất** (router JSON; `GEMINI_MODEL` để đổi) | Tự đáp câu đơn giản (`direct`), hoặc chọn tool + trích tham số (`tool`) | Tool tự format câu nói nên không tốn call tổng hợp thứ 2; `thinking_budget=0`, `max_output_tokens=300` |
| Tool (**61 cái**) | Local + API free không key (`voice/qa_tools.py::TOOL_FUNCS`) | Giờ/ngày, thời tiết Open-Meteo (+dự báo, mưa, UV, nắng/mưa), tiền tệ, đơn vị, tin tức (World Bank/USGS/thiên tai), trạng thái máy (CPU/RAM/disk/GPU/UpTime/mạng/port), file whitelist (txt/json/csv), **số liệu tracking trực tiếp** (số người, điểm danh, camera/FPS, STT latency...), chào/chụp qua pipeline (`request_greet`, `take_snapshot`) | Câu chào / wake hụt / STT rỗng không chạm API |
| TTS | **ZeroTTS** giọng `maichi` (CUDA, offline sau lần đầu tải weights ~900MB) | Đọc đáp án; filler + câu chào tạo sẵn WAV để phát ngay | `scripts/build_greeting_wavs.py` sinh `output/voice_greetings/` (+`manifest.json`); filler `output/voice_fillers/`: `ready/listening/thinking/got_it/missed.wav` (+`beep.wav`) |
| Loa bản cam | P2P VisualTalk qua `ImouP2PTalkOutput` | Phát đáp án ra loa camera | **Persistent tunnel** (bắt tay cloud 1 lần, giữ bằng heartbeat) + **cache AAC** + **preload AAC `listening.wav` ở warmup**; mỗi câu chào = 1 session; cấu hình `p2p_channel[_b]`, `timeout 20s`, `attempts 2`; chỉ cần `IMOU_DEVICE_ID` + `IMOU_CAMERA_PASSWORD` |
| Loa bản local | paplay → aplay → ffplay | Phát ra loa laptop | `mark_speaker_busy()` trước khi phát + nghỉ `tail_s 0.3s` sau phát cho vang tắt hẳn (chống echo) |
| Half-duplex | Mic mở **sau** khi loa dứt | Chống hú | P2P: phát xong mới mở mic lệnh; local: bỏ frame + hủy đoạn đang ghi khi `speaker_busy()` |

Chế độ **Trò chuyện** (chỉ bản local): câu có "lâu dài" + ngữ cảnh tâm sự
(`tam su`/`tro chuyen`/`noi chuyen`) → xin xác nhận có/không (treo vô thời
hạn, im `confirm-idle 45s` thì nhắc "vẫn đang nghe") → nghe liên tục miễn gọi
tên, 1 câu rep 1 câu (im `chat-idle 60s` thì nhắc) → "dừng trò chuyện" về QA.

Module `voice_trigger` ("hello imou") hiện **tắt** (`voice_trigger_enabled:
false`); STT trong pipeline tracking chỉ phục vụ đo đạc/dump
(`output/voice_dumps/`), Hà Linh là luồng nghe duy nhất đang chạy.

---

## 4. Chống trùng lặp / spam / nhiễu

### 4.1. Chào hỏi (loa) — không bao giờ chào 2 lần 1 người 1 lúc

`VoiceGreeter`: hàng đợi ưu tiên (chủ động trước, tự động sau) + thread phát
riêng, inference không bao giờ block.

- Mỗi người chỉ có **1 lượt trong hàng đợi** (gộp theo người); đang phát thì
  người khác xếp hàng, cùng người thì bỏ qua.
- **Cooldown theo loa** (khác loa chào riêng): quen 10s, lạ 10s
  (`cooldown_s` / `unknown_cooldown_s`); sau mỗi lần phát im 5s với chính
  người đó (`proactive_quiet_s`, cả chào chủ động lẫn tự động).
- Lệnh chào quá **TTL 5s** (`max_queue_age_s`) thì hủy (khỏi phát trễ), đồng
  thời **trả cooldown** để lần kế được chào ngay; lỗi TTS/P2P cũng rollback
  tương tự.
- Khách lạ đứng chờ định danh: `cancel_pending_unknown()` hủy lượt "quý
  khách" đã xếp khi mặt vừa xác nhận tên → chỉ nghe 1 câu chào tên, không bị
  chào đúp "quý khách" rồi chào tên.

### 4.2. Nghe nhầm (mic) — thà miss còn hơn nhầm

- Cụm gọi dài **"Hà Linh ơi"** thay vì từ ngắn; "hà" đứng một mình không bao
  giờ wake (tránh "ha ha", "hả"); "linh" một mình chỉ wake khi đoạn ≤ 2 tiếng
  (`HA_LINH_MAX_TOKENS_SINGLE`); bản cam thêm fuzzy chịu sai phát âm
  ("hẹn linh ơi" vẫn trúng ở ngưỡng 0.80).
- STT có prompt bias cụm gọi (`WAKE_STT_PROMPT`) + cửa chất lượng nới riêng
  cho vòng wake (`no_speech 0.60 / logprob -1.10`) so với vòng lệnh
  (`0.40 / -0.90`); `tiny` bịa wake từ ồn thì VAD/SNR và luật token chặn lại.
- Vang loa lọt mic: cắt tiền tố echo của chính câu ack (`strip_ack_echo`),
  câu nào chỉ còn trơ wake thì đợi câu lệnh thật (không đốt quota Gemini).
- Wake hụt N lần có tiếng (`--wake-miss-n`, mặc định 3) mới nhắc "nghe không
  rõ" 1 câu (`missed.wav`); STT rỗng (tiếng quạt/va đập) chỉ in heartbeat,
  không đếm hụt.
- Mic tắt khi loa phát (cờ `speaker_busy.json` xuyên process + `tail_s`) và
  inhibit sau mỗi câu chào (= câu chào dài nhất + đuôi vang, cấu hình
  `voice_trigger_inhibit_s: 12s`).

### 4.3. ID người — không swap, không ghost

- Appearance (ReID body 0.70) quyết định chính, vị trí chỉ phụ (0.05) → 2
  người đi qua nhau không đổi ID cho nhau; reconnect thuần vị trí chỉ cho
  dịch chuyển rất nhỏ (0.06, trong 2s).
- Track yếu (`min_hits: 3`) chết yểu ở tentative, không vào DB/lịch sử.
- Trùng chân dung (2 track cùng mặt, tương tự ≥ 0.90) gộp về 1 GID chuẩn.
- Điểm danh cần **đồng thuận**: qua threshold + margin + cửa chất lượng (mặt
  đủ lớn, đủ nét, đủ thẳng) + debounce; `unknown_cooldown 0.6s` /
  `known_cooldown 30s` ở tầng điểm danh.

### 4.4. Dữ liệu — mất mạng không mất dữ liệu

- Ghi Supabase qua **WriteQueue async** (`store/`, `queue.db`): offline thì
  xếp hàng local, có mạng flush sau. Attendance/ngày cũng preload + cache RAM.
- `identity_state.db` giữ gallery qua restart; snapshot ngày mới không trộn
  ID hôm qua.

### 4.5. Quota Gemini — 1 request/vòng, không hơn

- Không còn call tổng hợp đáp án (tool tự format câu nói) → tool đơn giản
  tốn đúng 1 call; câu chào/wake hụt/STT rỗng không chạm API.
- Hết quota (429) thì báo loa + ngủ 60s tự thử lại (tối đa 3 vòng), không bỏ
  câu hỏi; think treo quá `--think-timeout-s` (90s) thì báo loa "nghĩ lâu
  quá" và về chờ; xử lý quá `--think-filler-s` (1.2s) thì phát `thinking.wav`
  cho đỡ tưởng chết.
- `--with-search` mặc định tắt (search grounding tốn quota nặng trên key free).

---

## 5. Dữ liệu, dashboard, entry points

- **Lưu trữ**: `output/` (video annotate, `voice_fillers/`, `voice_greetings/`,
  `voice_dumps/`, `snapshots/`, `qa_cache/` gồm đáp án + transcript +
  3 file cầu nối, `queue.db`, `identity_state.db`), `data/images` (gallery
  mặt), Supabase (person/attendance/events) khi có mạng.
- **Dashboard** (`cd dashboard && npm run dev`, chung cho cả 2 bản): Live MJPEG
  `:8765` (single-channel tự ẩn khung cam B) + điều khiển pipeline `:8766`
  (chọn bản camera/local ở nút Start; bản embedded thì Start/Stop bị từ chối
  để không tự kill). Voice legacy dùng 8767 để khỏi đụng port.
- **Script hay dùng**: `run_workstate[_local].py` (tracking + kèm voice),
  `halinh_assistant[_local].py` (voice rời),
  `build_greeting_wavs.py` (tạo WAV chào/filler), `diag_voice.py` (chẩn đoán bridge
  voice legacy bằng Chromium/Playwright), `calibrate_room.py`
  (đo ROI), `pipeline_supervisor.py`, `test_imou_p2p_voice.py`.
- **Tune khi phòng ồn / mic yếu**: `docs/TUNE_GUIDE.md` + các ngưỡng VAD/STT
  trong khối `voice:` của `config/default.yaml` (`voice_vad_threshold`,
  `voice_vad_floor_*`, `voice_snr_min_db`, `voice_min_seg_rms`,
  `voice_max_no_speech_prob`, `voice_min_avg_logprob`).
