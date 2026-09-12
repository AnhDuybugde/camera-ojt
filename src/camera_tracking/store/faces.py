"""Luu face/frame crop tot nhat theo ngay (local mirror cua Supabase Storage).

Layout:
  {root}/YYYY-MM-DD/known/{person_id}_G{gid}_{HHMMSS}_s{score:.2f}.jpg
  {root}/YYYY-MM-DD/unknown/{uid}_G{gid}_{HHMMSS}_s{score:.2f}.jpg

Chi ghi de khi diem moi cao hon (best-shot) + gioi han N file/owner/ngay.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def crop_score(face_score: float, sharpness: float) -> float:
    sharp_norm = min(1.0, max(0.0, face_score if face_score else 0.0))
    sharp_part = min(1.0, sharpness / 300.0)  # Laplacian var ~300 la kha net
    return 0.7 * max(0.0, face_score) + 0.3 * sharp_part if sharpness else sharp_norm


class FaceCropSaver:
    def __init__(
        self, root: str | Path, max_per_owner_day: int = 5,
        jpeg_quality: int = 80, max_side_px: int = 512,
    ) -> None:
        self.root = Path(root)
        self.max_per_owner_day = max(1, max_per_owner_day)
        self.jpeg_quality = jpeg_quality
        self.max_side_px = max_side_px
        self._best: dict[tuple[str, str], float] = {}

    def save_best_crop(
        self,
        *,
        day: str,
        owner: str,  # person_id hoac unknown uid
        known: bool,
        global_id: int,
        image_bgr: np.ndarray | None,
        face_score: float,
        sharpness: float,
        time_tag: str,  # HHMMSS
    ) -> str | None:
        if image_bgr is None or image_bgr.size == 0:
            return None
        score = crop_score(face_score, sharpness)
        key = (day, owner)
        if score <= self._best.get(key, -1.0):
            return None  # giu best-shot cu, khong spam file
        sub = "known" if known else "unknown"
        folder = self.root / day / sub
        folder.mkdir(parents=True, exist_ok=True)
        # Gioi han so file/owner/ngay: xoa file diem thap nhat khi vuot.
        existing = sorted(folder.glob(f"{owner}_*.jpg"))
        if len(existing) >= self.max_per_owner_day:
            # Van cho phep ghi de best (xoa file cu nhat truoc khi luu moi).
            try:
                existing[0].unlink()
            except OSError:
                pass
        safe_score = max(0.0, min(1.0, float(face_score)))
        fname = f"{owner}_G{global_id}_{time_tag}_s{safe_score:.2f}.jpg"
        path = folder / fname
        img = self._downscale(image_bgr)
        try:
            import cv2

            cv2.imwrite(
                str(path), img,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
            )
        except ImportError:
            return None
        self._best[key] = score
        return str(path)

    def _downscale(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        side = max(h, w)
        if side <= self.max_side_px:
            return img
        try:
            import cv2
        except ImportError:
            return img
        scale = self.max_side_px / float(side)
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def save_best_crop(*args, **kwargs) -> str | None:
    """Helper functional giu backward-compat cho import cu."""
    saver = kwargs.pop("saver", None)
    if saver is None:
        raise TypeError("save_best_crop() can 'saver: FaceCropSaver'")
    return saver.save_best_crop(*args, **kwargs)
