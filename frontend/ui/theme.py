"""Inject the Stitch-faithful stylesheet into the Streamlit app.

IMPORTANT: st.html(path) does NOT reliably inject CSS.
           Use st.markdown(<style>...</style>) instead.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

STYLE_PATH = Path(__file__).resolve().parents[1] / "assets" / "style.css"


def apply_theme() -> None:
    css = STYLE_PATH.read_text(encoding="utf-8")
    # Inject as a proper <style> block into the Streamlit page head.
    # This is the only reliable way to apply global CSS in Streamlit.
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
