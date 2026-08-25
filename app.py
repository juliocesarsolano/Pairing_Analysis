"""Entry point for the Pairing Analysis Streamlit application."""

from __future__ import annotations

import streamlit as st

from src.ui.page import render_app
from src.ui.theme import apply_premium_theme


st.set_page_config(
    page_title="Pairing Analysis",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_premium_theme()
render_app()
