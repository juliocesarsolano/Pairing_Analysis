"""Premium Streamlit styling and reusable visual components."""

from __future__ import annotations

import html
from typing import Iterable

import streamlit as st


CORPORATE_BLUE = "#03547C"
CORPORATE_BLUE_DARK = "#004967"
CORPORATE_GOLD = "#A39161"
CORPORATE_ORANGE = "#FDB813"
CORPORATE_GRAY = "#C7C8CA"
CORPORATE_SLATE = "#44546A"


def apply_premium_theme() -> None:
    """Apply the application corporate visual language."""
    st.markdown(
        """
        <style>
        :root {
            --pa-blue: #03547C;
            --pa-blue-dark: #004967;
            --pa-gold: #A39161;
            --pa-orange: #FDB813;
            --pa-gray: #C7C8CA;
            --pa-slate: #44546A;
            --pa-bg: #F5F8FA;
            --pa-border: #D9E2EC;
            --pa-text: #1F2D36;
            --pa-muted: #637083;
        }

        .stApp {
            background: linear-gradient(180deg, #F8FAFB 0%, #FFFFFF 24%, #FFFFFF 100%);
        }

        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2.2rem;
            max-width: 1550px;
        }

        h1, h2, h3, h4 {
            color: var(--pa-blue-dark);
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F2F6F8 0%, #FFFFFF 100%);
            border-right: 1px solid rgba(0, 84, 124, 0.13);
        }

        div[data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem;
        }

        .pa-hero {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 1.4rem;
            align-items: center;
            padding: 1.35rem 1.5rem;
            margin: 0 0 1.15rem 0;
            border: 1px solid rgba(0, 84, 124, 0.18);
            border-top: 5px solid var(--pa-blue);
            border-radius: 14px;
            background:
                radial-gradient(circle at 92% 10%, rgba(163,145,97,0.16), transparent 24%),
                linear-gradient(135deg, #FFFFFF 0%, #F3F8FA 100%);
            box-shadow: 0 12px 28px rgba(0, 45, 67, 0.10);
        }

        .pa-kicker {
            color: var(--pa-gold);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .pa-title {
            color: var(--pa-blue-dark);
            font-size: clamp(1.75rem, 2.6vw, 2.55rem);
            font-weight: 800;
            line-height: 1.06;
            letter-spacing: -0.025em;
        }

        .pa-subtitle {
            margin-top: 0.55rem;
            color: #465B68;
            font-size: 0.98rem;
            line-height: 1.5;
            max-width: 850px;
        }

        .pa-chips {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.55rem;
            max-width: 460px;
        }

        .pa-chip {
            min-width: 118px;
            padding: 0.55rem 0.70rem;
            border: 1px solid rgba(0, 84, 124, 0.16);
            border-radius: 9px;
            background: rgba(255,255,255,0.88);
            box-shadow: 0 3px 8px rgba(0,45,67,0.05);
        }

        .pa-chip-label {
            display: block;
            color: #6A7782;
            font-size: 0.65rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .pa-chip-value {
            display: block;
            margin-top: 0.10rem;
            color: var(--pa-blue-dark);
            font-size: 0.88rem;
            font-weight: 750;
        }

        .pa-side-banner {
            margin: 1rem 0 0.55rem 0;
            padding: 0.58rem 0.72rem;
            border-left: 4px solid var(--pa-gold);
            border-radius: 6px;
            background: linear-gradient(90deg, rgba(3,84,124,0.10), rgba(255,255,255,0.70));
        }

        .pa-side-kicker {
            color: #6C7881;
            font-size: 0.62rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .pa-side-title {
            color: var(--pa-blue-dark);
            font-size: 0.91rem;
            font-weight: 800;
            margin-top: 0.06rem;
        }

        .pa-section-card {
            padding: 1rem 1.05rem;
            border: 1px solid var(--pa-border);
            border-radius: 12px;
            background: #FFFFFF;
            box-shadow: 0 5px 16px rgba(0,45,67,0.05);
        }

        .pa-mode-note {
            margin: 0.35rem 0 0.65rem 0;
            padding: 0.70rem 0.78rem;
            border: 1px solid rgba(163,145,97,0.35);
            border-left: 4px solid var(--pa-gold);
            border-radius: 7px;
            background: #FBF9F3;
            color: #46545C;
            font-size: 0.78rem;
            line-height: 1.45;
        }

        .pa-kpi-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.80rem;
            margin: 0.35rem 0 1rem 0;
        }

        .pa-kpi-card {
            min-height: 5.9rem;
            border: 1px solid rgba(0,84,124,0.10);
            border-bottom: 4px solid var(--pa-gold);
            border-radius: 10px;
            padding: 0.78rem 0.72rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: #FFFFFF;
            box-shadow: 0 4px 12px rgba(0,45,67,0.05);
        }

        .pa-kpi-value {
            color: #202A31;
            font-size: clamp(1.30rem, 2vw, 1.85rem);
            line-height: 1.0;
            font-weight: 650;
            text-align: center;
        }

        .pa-kpi-label {
            margin-top: 0.45rem;
            color: #5F6D76;
            font-size: 0.76rem;
            font-weight: 650;
            text-align: center;
        }

        div[data-testid="stTabs"] {
            margin-top: 0.45rem;
        }

        div[data-testid="stTabs"] div[role="tablist"],
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            display: flex !important;
            gap: 0.60rem !important;
            padding: 0.43rem !important;
            border: 1px solid rgba(0,84,124,0.20) !important;
            border-radius: 12px !important;
            background: linear-gradient(180deg, #F4F8FA 0%, #EAF2F6 100%) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.95), 0 6px 16px rgba(0,45,67,0.07) !important;
        }

        div[data-testid="stTabs"] button[role="tab"],
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            min-height: 2.45rem !important;
            padding: 0.55rem 0.95rem !important;
            border: 1px solid rgba(0,84,124,0.18) !important;
            border-radius: 8px !important;
            background: #FFFFFF !important;
            color: var(--pa-blue-dark) !important;
            font-weight: 750 !important;
            box-shadow: 0 2px 6px rgba(0,45,67,0.06) !important;
        }

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
        div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
            border-color: var(--pa-blue-dark) !important;
            background: linear-gradient(135deg, #004967 0%, #006F98 100%) !important;
            color: #FFFFFF !important;
            box-shadow: 0 7px 16px rgba(0,73,103,0.22) !important;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
        div[data-testid="stTabs"] [data-baseweb="tab-border"] {
            display: none !important;
        }

        div[data-testid="stDownloadButton"] button,
        div[data-testid="stButton"] button {
            border-radius: 8px;
        }

        @media (max-width: 1050px) {
            .pa-hero { grid-template-columns: 1fr; }
            .pa-chips { justify-content: flex-start; max-width: none; }
            .pa-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        @media (max-width: 620px) {
            .pa-kpi-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(chips: Iterable[tuple[str, str]] | None = None) -> None:
    """Render the premium top banner."""
    chip_html = "".join(
        f'<div class="pa-chip"><span class="pa-chip-label">{html.escape(label)}</span>'
        f'<span class="pa-chip-value">{html.escape(value)}</span></div>'
        for label, value in (chips or [])
    )
    st.markdown(
        f"""
        <section class="pa-hero">
            <div>
                <div class="pa-kicker">Geostatistics · Spatial Data Analysis</div>
                <div class="pa-title">Pairing Analysis</div>
                <div class="pa-subtitle">
                    GETPAIRS-compatible spatial pairing and paired-sample statistical comparison
                    for geochemical, geological and mineral-resource datasets.
                </div>
            </div>
            <div class="pa-chips">{chip_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_about() -> None:
    """Render compact application metadata in the sidebar."""
    with st.sidebar.expander("About", expanded=False):
        st.markdown(
            """
            **Pairing Analysis**  
            GETPAIRS-compatible spatial pairing and paired-sample statistical analysis.

            **Author:** Julio Solano  
            **Version:** 4.1
            """
        )


def sidebar_banner(kicker: str, title: str) -> None:
    """Render a compact corporate section banner in the sidebar."""
    st.markdown(
        f"""
        <div class="pa-side-banner">
            <div class="pa-side-kicker">{html.escape(kicker)}</div>
            <div class="pa-side-title">{html.escape(title)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pairing_mode_note(keep_closest: bool) -> None:
    """Explain the selected GETPAIRS mode in plain language."""
    if keep_closest:
        text = (
            "<b>Nearest neighbor only:</b> each reference sample contributes at most one pair — "
            "the closest comparison sample inside the search radius. The same comparison sample "
            "may still be reused by different reference samples."
        )
    else:
        text = (
            "<b>All neighbors within radius:</b> every comparison sample inside the search radius "
            "is retained. One reference sample can therefore generate several pair rows."
        )
    st.markdown(f'<div class="pa-mode-note">{text}</div>', unsafe_allow_html=True)


def render_kpi_cards(cards: list[tuple[str, str]]) -> None:
    """Render compact report-grade KPI cards."""
    html_cards = "".join(
        f'<div class="pa-kpi-card"><div class="pa-kpi-value">{html.escape(value)}</div>'
        f'<div class="pa-kpi-label">{html.escape(label)}</div></div>'
        for label, value in cards
    )
    st.markdown(f'<div class="pa-kpi-grid">{html_cards}</div>', unsafe_allow_html=True)
