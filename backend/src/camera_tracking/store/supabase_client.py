"""Supabase client (optional): bat khi .env co SUPABASE_URL + KEY.

- Backend pipeline dung SERVICE_KEY (ghi DB + upload Storage).
- Dashboard dung ANON_KEY + RLS (xem file supabase/schema.sql).
- Thieu env hoac thieu lib -> che do local-only, moi call tra False va
  caller day vao WriteQueue de flush sau.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class SupabaseSettings:
    url: str = ""
    service_key: str = ""
    anon_key: str = ""
    enabled: bool = False

    @classmethod
    def from_env(cls, enabled_flag: bool = False) -> SupabaseSettings:
        url = os.getenv("SUPABASE_URL", "").strip()
        service_key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
        anon_key = os.getenv("SUPABASE_ANON_KEY", "").strip()
        enabled = bool(enabled_flag and url and service_key)
        return cls(url=url, service_key=service_key, anon_key=anon_key, enabled=enabled)


class SupabaseStore:
    """Wrap nhe quanh supabase-py. Lazy import de may chua cai van chay."""

    def __init__(self, settings: SupabaseSettings | None = None) -> None:
        self.settings = settings or SupabaseSettings.from_env(False)
        self._client = None
        self._error: str | None = None

    @property
    def ready(self) -> bool:
        return bool(self.settings.enabled) and self._error is None

    @property
    def last_error(self) -> str | None:
        return self._error

    def connect(self) -> bool:
        if not self.settings.enabled:
            return False
        if self._client is not None:
            return True
        try:
            from supabase import create_client
        except ImportError as error:
            self._error = (
                f"supabase chua cai ({error}). Chay: pip install supabase"
            )
            return False
        try:
            self._client = create_client(self.settings.url, self.settings.service_key)
            return True
        except Exception as error:  # noqa: BLE001
            self._error = str(error)
            return False

    def reassign_face_crops(self, day: str, old_owner: str, person_id: str) -> bool:
        """Gop best-shot cua unknown owner ve person da diem danh (reconcile)."""
        if not self.connect():
            return False
        try:
            self._client.table("face_crops").update(
                {"owner_key": person_id, "is_known": True}
            ).eq("date", day).eq("owner_key", old_owner).execute()
            return True
        except Exception as error:  # noqa: BLE001
            self._error = str(error)
            return False

    # -- upserts --
    def upsert_person(self, row: dict) -> bool:
        if not self.connect():
            return False
        try:
            self._client.table("persons").upsert(
                row, on_conflict="person_id"
            ).execute()
            return True
        except Exception as error:  # noqa: BLE001
            self._error = str(error)
            return False

    def upsert_attendance(self, row: dict) -> bool:
        """row theo schema attendance_daily (PK date+person_id)."""
        if not self.connect():
            return False
        try:
            self._client.table("attendance_daily").upsert(
                row, on_conflict="date,person_id"
            ).execute()
            return True
        except Exception as error:  # noqa: BLE001
            self._error = str(error)
            return False

    def upsert_room_status(self, row: dict) -> bool:
        if not self.connect():
            return False
        try:
            self._client.table("room_status_daily").upsert(
                row, on_conflict="date,global_id"
            ).execute()
            return True
        except Exception as error:  # noqa: BLE001
            self._error = str(error)
            return False

    def upsert_current_state(self, row: dict) -> bool:
        """Update the durable employee-centric current state projection."""
        if not self.connect():
            return False
        try:
            self._client.table("employee_current_state").upsert(
                row, on_conflict="employee_id"
            ).execute()
            return True
        except Exception as error:  # noqa: BLE001
            self._error = str(error)
            return False

    def insert_event(self, row: dict) -> bool:
        if not self.connect():
            return False
        try:
            self._client.table("room_events").insert(row).execute()
            return True
        except Exception as error:  # noqa: BLE001
            self._error = str(error)
            return False

    def insert_attendance_audit(self, row: dict) -> bool:
        if not self.connect():
            return False
        try:
            self._client.table("attendance_audit_log").insert(row).execute()
            return True
        except Exception as error:  # noqa: BLE001
            self._error = str(error)
            return False

    def fetch_attendance_day(self, day: str) -> list[dict]:
        """Lấy attendance đã tick trong ngày để preload sau restart.

        Offline/thiếu env/lib -> trả [] (caller giữ behavior RAM-only).
        """
        if not self.connect():
            return []
        try:
            query = self._client.table("attendance_daily")
            try:
                response = (
                    query.select(
                        "date,person_id,person_name,global_id,check_in_at,"
                        "check_out_at,face_score,liveness_score,"
                        "verification_method,automatic,status"
                    )
                    .eq("date", day)
                    .execute()
                )
            except Exception:
                # Rolling deployment: stay compatible until the V2 migration
                # is applied, while still preventing duplicate check-ins.
                response = (
                    self._client.table("attendance_daily")
                    .select(
                        "date,person_id,person_name,global_id,check_in_at,face_score"
                    )
                    .eq("date", day)
                    .execute()
                )
            return list(getattr(response, "data", None) or [])
        except Exception as error:  # noqa: BLE001
            self._error = str(error)
            return []

    def upload_face_crop(self, local_path: str, storage_path: str) -> bool:
        """Upload len bucket 'face-crops'. storage_path dang YYYY-MM-DD/..."""
        if not self.connect():
            return False
        try:
            with open(local_path, "rb") as fh:
                self._client.storage.from_("face-crops").upload(
                    storage_path, fh,
                    file_options={"content-type": "image/jpeg", "upsert": "true"},
                )
            return True
        except Exception as error:  # noqa: BLE001
            self._error = str(error)
            return False
