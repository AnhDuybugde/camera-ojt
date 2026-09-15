# Imou AudioTalk MVP

Luồng đã triển khai:

`wave ổn định -> identity rõ ràng -> TTS MP3 cache -> localhost bridge -> Imou WebSDK AudioTalk -> loa camera`

- Employee đã match: `Xin chào <display_name>`.
- Face worker đã kết luận unknown: `Quen biết gì mà chào`.
- Chưa có kết quả face: không phát gì để tránh chào sai.
- Cooldown mặc định 60 giây theo employee; unknown cooldown theo Global ID.
- Một AudioTalk tại một thời điểm. Lời chào chờ quá 5 giây bị bỏ.
- Edge/Chrome được mở với `--mute-audio`; file TTS được inject vào microphone ảo của WebSDK và không phát qua loa PC.

## Chuẩn bị

1. Cài dependency:

   ```powershell
   pip install -r requirements.txt
   ```

2. Imou WebSDK hiện đã được đặt ở `output/imou_websdk` (Git bỏ qua thư mục
   `output`). Muốn dùng vị trí khác, đặt `IMOU_WEBSDK_DIR`; thư mục đó phải
   chứa ít nhất `imou-player.js` và thư mục `WasmLib`.

3. Điền `.env` theo `.env.example`: `IMOU_APP_ID`, `IMOU_APP_SECRET`,
   `IMOU_DEVICE_ID`, `IMOU_DEVICE_CODE`, `IMOU_WEBSDK_DIR`. Device code là
   safety/device code trên tem hoặc cấu hình camera, không phải mật khẩu tài
   khoản Imou Life. Mặc định dùng data center `sg` và channel `0`.

## Test một câu

Camera cần online và không có phiên intercom khác đang giữ AudioTalk:

```powershell
python scripts/test_imou_voice.py --text "Xin chào, đây là thử loa camera"
```

Một cửa sổ app Edge/Chrome nhỏ sẽ được mở để giữ WebSDK. Khi thành công CLI in
`Camera playback: completed`. Nếu token hết hạn, bridge refresh kit token và
nạp lại WebSDK. Mã lỗi AudioTalk được trả về CLI thay vì làm chết pipeline.

## Chạy cùng detection

```powershell
python scripts/run_workstate.py --greet
```

`voice.backend` trong `config/default.yaml` mặc định là `imou_web`. Backend
`local` vẫn tồn tại để debug bằng ffplay, nhưng luồng production phát qua loa
camera. TTS cần Internet lần đầu cho mỗi câu; sau đó dùng MP3 trong
`output/voice_cache`.

Camera hiện đang offline nên unit/integration test chỉ xác nhận logic HTTP,
token cache, serialization và ACK. Bài test một câu ở trên là bước xác nhận
phần cứng cuối cùng khi camera online.
