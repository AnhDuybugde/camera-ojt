# Floor-plane calibration

Camera gan tuong tao perspective distortion. Vi vay trajectory theo pixel khong the
dung truc tiep de so san van toc, khoang cach hay mat do giua cac khu vuc trong phong.

## Chuan bi

1. Co dinh camera tai vi tri su dung that; khong thay doi zoom sau khi calibration.
2. Chon it nhat bon diem dong phang, de nhin thay tren san. Uu tien cac goc tao thanh
   mot vung lon, tranh bon diem gan thang hang.
3. Do toa do `(x, y)` cua tung diem tren san theo met, voi mot goc phong lam `(0, 0)`.
4. Lay toa do pixel cua dung cac diem do tren mot frame camera.
5. Ghi hai danh sach cung thu tu vao `config/default.yaml`.

Vi du:

```yaml
calibration:
  image_points: [[240, 700], [1040, 700], [820, 260], [460, 260]]
  floor_points: [[0.0, 0.0], [8.0, 0.0], [8.0, 6.0], [0.0, 6.0]]
```

## Kiem tra

- Trajectory phai nam trong kich thuoc `floor_width_m` x `floor_height_m`.
- Nguoi di thang tren san phai tao trajectory gan thang trong toa do san.
- Di qua counting line mot lan phai chi tang mot trong hai bien `entries` hoac `exits`.
- Neu camera bi dich chuyen, can calibration lai.

Homography gia dinh mat san phang. Bac, ram doc manh hay nhieu cao do san can calibration
theo tung mat phang hoac mot mo hinh camera 3D day du hon.
