"""Shared Streamlit service container."""
from __future__ import annotations

import streamlit as st

from attendance.attendance_service import AttendanceService
from attendance.daily_service import DailyAttendanceService, DailyAttendanceWorker
from attendance.admin_service import AttendanceAdminService
from auth.service import AuthService
from database.db import Database
from face.detector import FaceDetector
from face.recognizer import FaceRecognizer
from google_sheets.sheets_client import SheetsClient
from google_sheets.sync_service import SyncService, SyncWorker


@st.cache_resource
def get_db(schema_version: int = 4) -> Database:
    # The version participates in Streamlit's cache key after additive migrations.
    del schema_version
    return Database()


@st.cache_resource
def get_detector() -> FaceDetector:
    return FaceDetector()


@st.cache_resource
def get_recognizer() -> FaceRecognizer:
    return FaceRecognizer(get_db(4))


@st.cache_resource
def get_attendance_service(runtime_version: int = 4) -> AttendanceService:
    # Refresh the service when attendance validation rules change.
    del runtime_version
    return AttendanceService(get_db(4), on_change=get_sync_worker().request_sync)


@st.cache_resource
def get_daily_attendance_service() -> DailyAttendanceService:
    return DailyAttendanceService(get_db(4), on_change=get_sync_worker().request_sync)


@st.cache_resource
def get_daily_attendance_worker() -> DailyAttendanceWorker:
    return DailyAttendanceWorker(get_daily_attendance_service()).start()


@st.cache_resource
def get_sync_service() -> SyncService:
    return SyncService(get_db(4), SheetsClient())


@st.cache_resource
def get_sync_worker() -> SyncWorker:
    return SyncWorker(get_sync_service()).start()


@st.cache_resource
def get_auth_service(runtime_version: int = 1) -> AuthService:
    del runtime_version
    return AuthService(get_db(4))


@st.cache_resource
def get_attendance_admin_service(runtime_version: int = 1) -> AttendanceAdminService:
    del runtime_version
    return AttendanceAdminService(get_db(4), on_change=get_sync_worker().request_sync)
