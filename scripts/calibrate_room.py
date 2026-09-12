"""Click-to-calibrate: no tape measure needed.

Opens the camera view (resized to config.camera size, the exact frame the
pipeline sees), you click, it writes a ready-to-run location config:

  1. Click >= 4 floor reference points. Type meters per point, OR use
     --tile 0.6 and just count floor tiles (col, row) in the terminal.
  2. Per desk: type a name, click its core polygon, click its extended
     polygon. Empty name = done.
  3. Saves --out (e.g. config/locations/roomA.yaml), then run:
       python scripts\\run_workstate.py --config <out> [--display]

Camera moved but same room? Re-run with --keep-ws <old-file>: desks (in
meters) are kept, you only re-click the 4 calibration points.

Keys in the window: click = add point, U = undo, D = done this polygon.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

from camera_tracking.calibration import (
    build_location_config,
    detect_tile_grid,
    topdown_pairs,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click calibration for rooms.")
    parser.add_argument("--base", type=Path, default=Path("config/default.yaml"),
                        help="Base config to copy (frame size + the rest).")
    parser.add_argument("--out", type=Path, default=Path("config/locations/room.yaml"),
                        help="Where to write the location config.")
    parser.add_argument("--source", default=None,
                        help="Camera/frame source (default: IMOU A sub stream, else webcam).")
    parser.add_argument("--image", type=Path, default=None,
                        help="Calibrate from a saved camera frame instead of live.")
    parser.add_argument("--tile", type=float, default=0.0,
                        help="Floor tile size in meters (e.g. 0.6): then just count tiles.")
    parser.add_argument("--auto-tiles", action="store_true",
                        help="Auto-detect tile grid (needs --tile); fallback to clicks.")
    parser.add_argument("--topdown", action="store_true",
                        help="Near-nadir camera: click only 2 points --tile meters apart.")
    parser.add_argument("--keep-ws", type=Path, default=None,
                        help="Keep workstations from this file, redo calibration only.")
    return parser.parse_args()


def grab_frame(source: str | int | None, image: Path | None):
    if image is not None:
        frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"Khong doc duoc anh: {image}")
        return frame
    candidates: list = []
    if source is not None:
        candidates.append(source)
    else:
        ip = os.getenv("IMOU_IP", "")
        user = os.getenv("IMOU_USER", "")
        password = os.getenv("IMOU_PASSWORD", "")
        if ip and user and password:
            from urllib.parse import quote

            candidates.append(
                f"rtsp://{user}:{quote(password, safe='')}@{ip}:554"
                f"/cam/realmonitor?channel=1&subtype=1")
        candidates.append(0)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    for cand in candidates:
        try:
            src = int(cand) if str(cand).isdigit() else cand
        except ValueError:
            src = cand
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) \
            if not isinstance(src, int) else cv2.VideoCapture(src, cv2.CAP_DSHOW)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok and frame is not None:
            print(f"Lay hinh tu: {cand}")
            return frame
    raise SystemExit("Khong mo duoc camera/anh. Thu --image <frame.jpg>.")


class ClickState:
    def __init__(self) -> None:
        self.points: list[tuple[int, int]] = []
        self.done_polys: list[tuple[list, tuple[int, int, int], str]] = []
        self.mode_text = ""


def main() -> None:
    args = parse_args()
    with open(args.base, encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}
    width = int(base.get("camera", {}).get("frame_width", 1280))
    height = int(base.get("camera", {}).get("frame_height", 720))

    frame = grab_frame(args.source, args.image)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    canvas = frame.copy()

    state = ClickState()

    def redraw() -> None:
        img = canvas.copy()
        for poly, color, label in state.done_polys:
            pts = np.asarray(poly, dtype=np.int32)
            if len(pts) >= 3:
                cv2.polylines(img, [pts], True, color, 2)
            for i, (x, y) in enumerate(poly):
                cv2.circle(img, (int(x), int(y)), 4, color, -1)
            if poly:
                cv2.putText(img, label, (int(poly[0][0]) + 6, int(poly[0][1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for i, (x, y) in enumerate(state.points):
            cv2.circle(img, (x, y), 5, (255, 255, 0), -1)
            cv2.putText(img, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if len(state.points) >= 2:
            cv2.polylines(img, [np.asarray(state.points, np.int32)],
                          False, (255, 255, 0), 2)
        cv2.putText(img, state.mode_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, "click=add  U=undo  D=done", (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("calibrate (nhan D khi xong, U de undo)", img)

    def collect(prompt: str, minimum: int) -> list[tuple[int, int]]:
        state.points = []
        state.mode_text = prompt
        redraw()
        print(f"\n{prompt}: click chuot, xong nhan D (>= {minimum} diem)")

        def on_mouse(event, x, y, _flags, _param) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                state.points.append((x, y))
                redraw()

        cv2.setMouseCallback("calibrate (nhan D khi xong, U de undo)", on_mouse)
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("u"), ord("U")):
                if state.points:
                    state.points.pop()
                    redraw()
            elif key in (ord("d"), ord("D")):
                if len(state.points) >= minimum:
                    return list(state.points)
                print(f"Can it nhat {minimum} diem (dang co {len(state.points)}).")

    cv2.namedWindow("calibrate (nhan D khi xong, U de undo)", cv2.WINDOW_NORMAL)

    import numpy as _np

    img_pts: list = []
    floor_pts: list = []
    homography = None

    def compute_homography():
        src = _np.asarray(img_pts, dtype=_np.float32)
        dst = _np.asarray(floor_pts, dtype=_np.float32)
        matrix, _ = cv2.findHomography(src, dst)
        if matrix is None:
            raise SystemExit("Khong dung duoc homography tu cac diem nay.")
        return matrix

    # --- Step 1a: topdown shortcut (2 clicks, near-nadir camera) ---
    if args.topdown:
        if not args.tile or args.tile <= 0:
            raise SystemExit("--topdown can --tile <co vien gach, vd 0.6>.")
        pair = collect(f"Topdown: click 2 diem cach nhau {args.tile} m "
                       f"(vd 2 dau 1 vien gach)", 2)
        img_pts, floor_pts = topdown_pairs(
            (float(pair[0][0]), float(pair[0][1])),
            (float(pair[1][0]), float(pair[1][1])), args.tile)
        img_pts = [list(p) for p in img_pts]
        floor_pts = [list(p) for p in floor_pts]
        print("Topdown: dung ti le truc tiep (gia dinh camera khong xoay).")

    # --- Step 1b: auto tile grid (0 click calib, can --tile) ---
    if homography is None and not img_pts and args.auto_tiles:
        if not args.tile or args.tile <= 0:
            raise SystemExit("--auto-tiles can --tile <vd 0.6>.")
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found = detect_tile_grid(gray, args.tile)
            print(f"Tu thay {len(found)} giao diem ron gach.")
            preview = frame.copy()
            for (x, y), _ in found[::max(1, len(found) // 200)]:
                cv2.circle(preview, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.imshow("calibrate (nhan D khi xong, U de undo)", preview)
            choice = input("Chap nhan luoi gach? [Y = dong y / M = click tay]: "
                           ).strip().lower()
            if choice in ("", "y", "yes"):
                img_pts = [[x, y] for (x, y), _ in found]
                floor_pts = [[fx, fy] for _, (fx, fy) in found]
                homography = compute_homography()
                state.done_polys.append (
                    ([(x, y) for x, y in img_pts], (0, 255, 0), "tiles"))
                redraw()
        except ValueError as error:
            print(f"Auto-tile that bai ({error}) -> chuyen sang click tay.")

    # --- Step 1c: manual clicks (fallback / mac dinh) ---
    if homography is None and not img_pts:
        img_pts = [[float(x), float(y)] for x, y in
                   collect("Buoc 1: click >= 4 diem san tham chieu", 4)]
        floor_pts = []
        if args.tile and args.tile > 0:
            print(f"Gach {args.tile} m: KHONG go met, chi dem o gach.")
            print("  Chon 1 goc phong lam goc (0,0). Cot dem sang phai, hang dem")
            print("  xuong duoi. Vd diem roi vao o cot 3 hang 5 thi go: 3 5")
            print("  (tool tu doi ra met: 3x0.6=1.8m, 5x0.6=3.0m).")
            for i, _ in enumerate(img_pts):
                while True:
                    try:
                        col, row = input(f"  diem {i + 1} [cot hang]: ").split()
                        c, r = int(col), int(row)
                        floor_pts.append([c * args.tile, r * args.tile])
                        break
                    except ValueError:
                        print("  Nhap 2 so nguyen, vd: 3 5")
        else:
            print("Go toa do met cho tung diem ('x y', vd '1.2 0.6'). "
                  "Uoc luong buoc chan cung duoc (sai vai chuc cm van OK).")
            for i, _ in enumerate(img_pts):
                while True:
                    try:
                        x, y = input(f"  diem {i + 1} [x y met]: ").split()
                        floor_pts.append([float(x), float(y)])
                        break
                    except ValueError:
                        print("  Nhap 2 so, vd: 1.2 0.6")
        homography = compute_homography()
        state.done_polys.append (
            ([(x, y) for x, y in img_pts], (255, 0, 255), "calib"))
        redraw()

    def to_floor(px: list[tuple[int, int]]) -> list[list[float]]:
        pts = _np.asarray([[list(p) for p in px]], dtype=_np.float32)
        mapped = cv2.perspectiveTransform(pts, homography)[0]
        return [[float(x), float(y)] for x, y in mapped]

    # --- Step 2: workstations ---
    workstations: list[dict] = []
    if args.keep_ws is not None:
        with open(args.keep_ws, encoding="utf-8") as fh:
            kept = (yaml.safe_load(fh) or {}).get("workstations", []) or []
        workstations = kept
        print(f"Giu {len(workstations)} workstation tu {args.keep_ws} "
              f"(khong can ve lai).")
    else:
        while True:
            name = input("\nTen ban/ghe (Enter de xong): ").strip()
            if not name:
                break
            core = collect(f"Ban {name}: click CORE (vung ban/ghe)",
                           3)
            state.done_polys.append (
                ([(x, y) for x, y in core], (0, 255, 0), f"{name}:core"))
            redraw()
            extended = collect(f"Ban {name}: click EXTENDED (vung lan can)",
                               3)
            workstations.append({"name": name,
                                 "core": to_floor(core),
                                 "extended": to_floor(extended)})
            state.done_polys.append (
                ([(x, y) for x, y in extended], (0, 255, 255),
                 f"{name}:ext"))
            redraw()
            print(f"  -> {name}: core/extended (met) da ghi nhan.")

    config = build_location_config(
        base,
        [[float(x), float(y)] for x, y in img_pts],
        floor_pts,
        workstations,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)
    cv2.destroyAllWindows()
    print(f"\nXong! Da ghi {args.out} "
          f"({len(img_pts)} diem calib, {len(workstations)} workstation).")
    print(f"Chay: python scripts\\run_workstate.py --config {args.out} --display")


if __name__ == "__main__":
    main()
