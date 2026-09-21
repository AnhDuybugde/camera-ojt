"""Persistence daily: cache chong spam + luu face crop + queue offline + Supabase.

- DailyStateCache: attendance upsert 1 lan/ngay; room_status update khi doi
  label; flip in_room coalesce toi da 1 lan/gio (trung gian chi vao events).
- FaceCropSaver: moi (date, owner) giu best-shot + toi da N crops/ngay.
- WriteQueue: SQLite hang doi khi mat mang / Supabase chua cau hinh.
"""
from camera_tracking.store.daily import DailyStateCache, RoomStatusRow, throttle_ok
from camera_tracking.store.faces import FaceCropSaver, crop_score, save_best_crop
from camera_tracking.store.queue import WriteQueue, WriteQueueWorker

__all__ = [
    "DailyStateCache",
    "FaceCropSaver",
    "RoomStatusRow",
    "WriteQueue",
    "WriteQueueWorker",
    "crop_score",
    "save_best_crop",
    "throttle_ok",
]
