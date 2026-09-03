# Camera Tracking

Du an theo doi luu luong nguoi trong phong, mat do va luong di chuyen dua tren camera gan tren tuong.

## Muc tieu

- Dem so nguoi ra vao va tong so nguoi co mat trong khung hinh.
- Uoc tinh mat do nguoi theo tung vung trong phong.
- Theo doi luong di chuyen va huong di chuyen theo thoi gian.
- Ho tro nguon video tu camera IP, file video mau hoac webcam.

## Cau truc thu muc

```text
camera-ojt/
├── config/                 # Cau hinh camera, model, tracking, vung quan sat
├── data/                   # Du lieu cuc bo, khong commit file lon
│   ├── raw/                # Video/anh goc
│   ├── processed/          # Ket qua xu ly trung gian
│   └── samples/            # Mau nho de test
├── docs/                   # Tai lieu thiet ke, yeu cau, ghi chu nghiep vu
├── models/                 # Trong so model cuc bo, khong commit file lon
├── notebooks/              # Notebook thu nghiem
├── output/                 # Ket qua chay: report, anh, video annotate
├── scripts/                # Script CLI de chay pipeline
├── src/
│   └── camera_tracking/
│       ├── analytics/      # Tinh mat do, luu luong, heatmap, trajectory
│       ├── camera/         # Ket noi va doc stream camera
│       ├── detection/      # Phat hien nguoi
│       ├── tracking/       # Gan ID va theo doi doi tuong
│       ├── visualization/  # Ve overlay, dashboard output
│       └── utils/          # Ham tien ich
└── tests/                  # Unit/integration tests
```

## Khoi chay nhanh

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\run_pipeline.py --config config\default.yaml
```

## Ghi chu

Thu muc `data/`, `models/` va `output/` chi giu file `.gitkeep` trong Git. Video, model weights va ket qua sinh ra nen luu cuc bo hoac dua len storage rieng.
