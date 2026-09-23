"""Streamlit API clients; no database or background worker lives in the UI."""
import streamlit as st
from integration.backend import BackendWorker, RemoteDetector, RemoteRecognizer, RemoteService


@st.cache_resource
def get_db(schema_version=4):
    return RemoteService("employees")


@st.cache_resource
def get_detector():
    return RemoteDetector()


@st.cache_resource
def get_recognizer():
    return RemoteRecognizer()


@st.cache_resource
def get_attendance_service(runtime_version=4):
    return RemoteService("attendance")


@st.cache_resource
def get_sync_service():
    return RemoteService("sync")


@st.cache_resource
def get_sync_worker():
    return BackendWorker()


@st.cache_resource
def get_auth_service(runtime_version=1):
    return RemoteService("auth")


@st.cache_resource
def get_attendance_admin_service(runtime_version=1):
    return RemoteService("attendance")
