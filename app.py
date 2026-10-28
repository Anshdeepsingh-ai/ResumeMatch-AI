"""
ResumeMatch AI - Main Streamlit Application Entrypoint.
Course: CSE472 Deep Learning for NLP
Milestone: Day 1 Project Foundation
"""

import streamlit as st

from config.settings import settings
from src.utils.logger import get_logger
from ui.dashboard import render_ranking_dashboard
from ui.landing import apply_custom_styles, render_landing_page, render_sidebar

# Initialize logger
logger = get_logger("app")


def main() -> None:
    """Initialize and run the Streamlit application."""
    logger.info("Initializing %s (v%s) in %s mode", settings.APP_NAME, settings.VERSION, settings.APP_ENV)

    # Configure Streamlit page layout
    st.set_page_config(
        page_title=f"{settings.APP_NAME} - Semantic Resume Matching",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()
    render_sidebar()

    # Tabbed navigation between Overview and Candidate Ranking & Analytics
    tab_overview, tab_ranking = st.tabs([
        "💡 Project Overview",
        "🏆 Candidate Ranking & Analytics",
    ])

    with tab_overview:
        render_landing_page(include_sidebar=False)

    with tab_ranking:
        render_ranking_dashboard()

    logger.info("Application view successfully rendered.")


if __name__ == "__main__":
    main()
