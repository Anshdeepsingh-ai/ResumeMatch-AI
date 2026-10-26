"""
ResumeMatch AI - Main Streamlit Application Entrypoint.
Course: CSE472 Deep Learning for NLP
Milestone: Day 1 Project Foundation
"""

import streamlit as st

from config.settings import settings
from src.utils.logger import get_logger
from ui.landing import render_landing_page

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

    # Render modular landing page
    render_landing_page()
    logger.info("Landing page successfully rendered.")


if __name__ == "__main__":
    main()
