# Team workflow — tránh conflict

## Ranh giới module

| Thành viên | Được sửa chính | Không sửa trực tiếp |
|---|---|---|
| UI | `frontend/` | thuật toán trong `backend/src/camera_tracking/` |
| Model | `backend/src/camera_tracking/{detection,tracking,face,workstate}/`, `backend/config/` | `audio/`, UI |
| Audio | `backend/src/camera_tracking/audio/`, `backend/src/camera_tracking/integration/`, `backend/scripts/run_be_xinh_bridge.py` | vòng lặp model và UI |

Các file root (`run.ps1`, contract `/status.json`, dependency files) là vùng dùng
chung; thay đổi cần được cả nhóm review.

## Nhánh đề xuất

```text
feature/ui-<ten-ngan>
feature/model-<ten-ngan>
feature/audio-<ten-ngan>
```

Trước khi tạo pull request:

```powershell
git switch main
git pull --ff-only origin main
git switch <feature-branch>
git merge main
```

Giải quyết conflict ngay trên feature branch, chạy test module của mình rồi mới
merge. Không dùng `git add .` khi chưa xem `git status`; không commit `.env`,
face dataset, log, model weights hoặc audio cache.

## Contract giữa ba phần

- Model cung cấp `GET /status.json`, `/cam_a.mjpg`, `/cam_b.mjpg`.
- UI chỉ đọc API/MJPEG, không import code model.
- Audio chỉ đọc người đã xác nhận từ `status.json`; không truy cập database UI.
- Khi đổi schema `status.json`, cập nhật adapter + test trong cùng pull request.
