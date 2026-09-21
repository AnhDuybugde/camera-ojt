# Loa camera qua P2P VisualTalk (backend `imou_p2p`)

Backend mac dinh moi cho luong **employee dung truoc camera + vay tay ->
`Xin chào <ten>`**, thay the backend cu `imou_web` (Chrome + WebSDK +
kitToken + virtual mic).

## Luong hoat dong

```text
Camera A/B -> YOLO person -> ByteTrack -> Global ID
  -> InsightFace (quen/la) + MediaPipe Hands WaveDetector (vay tay)
  -> VoiceGreeter (edge-tts vi-VN, cache mp3)
  -> ImouP2PTalkOutput
    -> ffmpeg: AAC-LC/ADTS mono 16 kHz
    -> DHAV interleaved audio frames
    -> visualtalk.xav talkback session
    -> DHP2P/PTCP relay tunnel (remote camera port 8086)
    -> loa camera
```

- Inference khong bao gio block: `VoiceGreeter` co queue + thread rieng,
  moi loi chao cooldown 60s/nguoi (`voice.cooldown_s`), wave cooldown 30s.
- Khong mo mic laptop, khong phat loa laptop, khong can browser.
- `200 OK` + `Sent N DHAV frames` = transport OK. Nghe duoc hay khong con
  phu thuoc firmware/volume/channel/Imou cloud (ghi nhan tu repo goc).

## Cau hinh

`.env` (chi 2 bien bat buoc):

```dotenv
IMOU_DEVICE_ID=your_camera_serial_number
IMOU_CAMERA_PASSWORD=your_camera_safety_code
```

Fallback password: `IMOU_CAMERA_PASSWORD` -> `IMOU_PASSWORD` ->
`IMOU_DEVICE_CODE` -> `IMOU_DEVICE_PASSWORD`. Nhieu camera IMOU dan dung
safety code lam luon RTSP password.

`config/default.yaml`:

```yaml
voice:
  enabled: false        # bat bang --greet
  backend: imou_p2p     # imou_web = legacy, local = ffplay tai may
  p2p_channel: 1
  p2p_timeout_s: 20.0
  p2p_attempts: 2
```

## WAV chao tao san bang ZeroTTS (khuyen nghi)

Moi cau chao la **1 file WAV duy nhat**, phat trong **1 VisualTalk session**
nen khong the ngat quang giua "Xin chào" va ten:

```powershell
python -m pip install -e ".[voice]"   # gom zerotts + soundfile
python scripts/build_greeting_wavs.py --list-voices
python scripts/build_greeting_wavs.py --voice maichi
```

Sinh ra `output/voice_greetings/` (gitignored, offline sau lan dau):
`unknown.wav` ("Xin chào quý khách") + 1 file/nhan vien
("Xin chào <ten>" lay tu `data/images/registry.json`) + `manifest.json`.
Pipeline (`--greet`) tu doc manifest: khop nguyen van cau chao thi dung WAV
co san, khong can mang/TTS runtime. Log `pregen=5` la nhan du 5 cau.

Doi cau unknown (vi du chi "Xin chào"): sua `voice.unknown_phrase` trong
`config/default.yaml` roi chay lai script (co `--force`, `--only`).

## Chay thu

```powershell
python -m pip install -e ".[voice]"
python scripts/test_imou_p2p_voice.py --greeting QuocNgoc   # thu loa, khong can pipeline
python scripts/test_imou_p2p_voice.py --greeting unknown
python scripts/test_imou_p2p_voice.py --text "Xin chào Duy"
python scripts/run_workstate.py --greet --display           # test gesture that
```

## Test gesture vay tay (checklist khi khong thay chao)

1. Dung gan camera 1-2m, nhin thang (de face match truoc), tay giơ cao
   ngang mat, vẫy **to + lien tuc 2-3 giay**.
2. Console phai co `Voice: imou_p2p, ... pregen=5`. Khi fire se in
   `[Wave][A] <ten> (G<n>)`; chua du thi in `dao chieu x/4 (vay to...ron chut)`;
   thay tay nhung chua thay mat thi in `chua nhan dien mat`.
3. Khong co dong `[Wave]` nao = tay qua nho/xa hoac nguoc sang (MediaPipe
   khong thay ban tay). Dang `WAVE` cung hien o dashboard (Live events).
4. Co `[Wave]` nhung khong nghe: xem loi `[Voice...]` (P2P/credentials/mang);
   `200 OK` + frames ma van câm thi do volume/firmware camera.
5. Chinh do nhay: `voice.wave_every_k` (nho = mau day, ton CPU),
   `voice.wave_window_s`, `voice.wave_min_reversals`.

File WAV co san (vi du tu test-sound-camera-imou
`audio/camera-test-vi.wav`):

```powershell
python scripts/test_imou_p2p_voice.py --audio .\audio\camera-test-vi.wav
```

## Vendor + attribution

- `src/camera_tracking/voice/p2p/imou_*.py` vendor tu
  https://github.com/AnhDuybugde/test-sound-camera-imou (one-shot WAV
  talkback, da test protocol-level voi Imou Ranger Dual Pro 10MP
  `IPC-S2XEP-10M0S`: camera nhan VisualTalk session `200 OK`).
- Protocol modules goc dan xuat tu public research
  https://github.com/home-assistant-tools/imou-life (minimal code path cho
  WAV talkback, khong gom RTSP/ONVIF/NetSDK/browser-OpenSDK/mic-loopback).
  Xem license/notice cua upstream truoc khi dung production.
- Wrapper moi: `src/camera_tracking/voice/p2p_talk.py`
  (`ImouP2PCredentials`, `send_audio_file`, `ImouP2PTalkOutput`).
- Legacy giu lai: `src/camera_tracking/voice/imou_bridge.py`,
  `scripts/test_imou_voice.py`, `scripts/diag_voice.py`.

## Bao mat

- Chi dung voi camera/tai khoan minh so huu hoac duoc uy quyen quan tri.
- Khong commit `.env`, serial, safety code, log/token, packet capture.
- Day la interoperability/research code, khong phai official Imou SDK.
  Firmware/cloud thay doi co the lam hong protocol.
