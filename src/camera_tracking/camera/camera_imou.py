import os
import cv2
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()

# Ép RTSP chạy bằng TCP để ổn định hơn
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

IP = os.getenv("IMOU_IP", "")
USER = os.getenv("IMOU_USER", "")
PASSWORD = os.getenv("IMOU_PASSWORD", "")

if not IP or not USER or not PASSWORD:
    raise ValueError(
        "Thiếu biến môi trường IMOU_IP / IMOU_USER / IMOU_PASSWORD. "
        "Hãy kiểm tra file .env"
    )

# Encode password phòng trường hợp có ký tự đặc biệt
password = quote(PASSWORD, safe="")

# 2 mắt camera
url1 = f"rtsp://{USER}:{password}@{IP}:554/cam/realmonitor?channel=1&subtype=1"
url2 = f"rtsp://{USER}:{password}@{IP}:554/cam/realmonitor?channel=2&subtype=1"

cap1 = cv2.VideoCapture(url1, cv2.CAP_FFMPEG)
cap2 = cv2.VideoCapture(url2, cv2.CAP_FFMPEG)

if not cap1.isOpened():
    print("Không mở được camera mắt 1")

if not cap2.isOpened():
    print("Không mở được camera mắt 2")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1:
        # Sau này xử lý Computer Vision ở đây
        # result = model(frame1)
        cv2.imshow("IMOU - Camera 1", frame1)

    if ret2:
        # Sau này xử lý Computer Vision ở đây
        cv2.imshow("IMOU - Camera 2", frame2)

    # Nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()