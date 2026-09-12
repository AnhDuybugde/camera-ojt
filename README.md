# Camera OJT — diem danh + trang thai phong 2 camera

YOLO26s -> ByteTrack -> Global ID xuyen 2 cam IMOU (ReID histogram) +
diem danh mat InsightFace `buffalo_s` o cam cua + dashboard Vite +
Supabase (free tier).

## Cai dat 1 lan

```powershell
python -m pip install -e ".[face,store]"
copy .env.example .env            # dien IMOU_* + SUPABASE_URL/KEY
cd dashboard; npm install; cd ..
```

Supabase (1 lan, SQL Editor): chay `supabase/schema.sql`, tao 2 bucket
`face-crops` + `enrolled-faces`, tao user admin roi chay:

```sql
insert into public.roles (user_id, role) values ('<uuid-admin>', 'admin');
```

Bo anh enroll vao `data/images/` (ten file = ID, vd `LeHoAnhDuy.jpg`).
Anh mat KHONG commit len git.

## Chay (2 command)

```powershell
# 1. Cam + model + stream (cua so 1)
python scripts\run_workstate.py --display

# 2. Website: Live 2 cam + diem danh + trang thai + admin (cua so 2)
cd dashboard; npm run dev   # http://localhost:5173
```

Live tren web doc qua `VITE_STREAM_URL` (mac dinh `http://localhost:8765`,
pipeline tu mo). Xem tu may khac: pipeline them `--stream-host 0.0.0.0`,
sua `VITE_STREAM_URL` thanh `http://<IP-may-cam>:8765`, restart `npm run dev`.

## Tinh chinh nhanh

| Nhu cau | Lam gi |
|---|---|
| Bot bao ao (ghost ID) | `--new-track-conf 0.5`, `--min-area 5000` |
| Nguoi that hien G cham | `--new-track-conf 0.3` |
| Roi cho bao Away nhanh/cham | `--move-ratio 0.10` / `0.20` (`0` = tat) |
| Mat kho khop | them anh enroll, hoac `face.match_threshold: 0.4` |
| Nhe CPU | `--imgsz 640` (mặc định 800, CUDA) |
| Tracking-only, tat mat | them `--no-face` |

Trang thai: ngoi yen = Working, roi khoi diem neo = Away (ca khi van
trong hinh), vang lau + thay o cam B = Out of office, quay lai =
Returning. Overlay/box mau theo trang thai: xanh la Working, vang
Away/Near seat, xanh duong Returning, do Out/Unknown. Ten hien ngay khi
khop mat (tick DB van debounce 2 hits/8s, cong don xuyen ID vo vun). Mat la cap `U-...`, tu gop
vao nguoi quen khi khop mat sau (log `[Reconcile]`). G non (< 5 hits)
khong ghi DB.

Ve san/workstation (khi co dinh camera): click tool thay vi do tay:

```powershell
python scripts\calibrate_room.py --out config\locations\roomA.yaml --tile 0.6 --auto-tiles
python scripts\run_workstate.py --config config\locations\roomA.yaml --display
```

Admin tren dashboard (can login): tick diem danh, dao in-room, chon
dong / xoa all lich su trong ngay (2 lop confirm), nut Start/Stop
pipeline (can chay `python scripts\pipeline_supervisor.py`).

## Test

```powershell
python -m pytest -q
```
