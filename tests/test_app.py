"""
Smoke tests and verification tests for ResumeMatch AI.
Course: CSE472 Deep Learning for NLP
Milestone: Day 1 Project Foundation
"""

import importlib
import logging
from pathlib import Path

import pytest
from config.settings import Settings, settings
from src.utils.logger import get_logger, setup_logger


class TestConfiguration:
    """Smoke tests for centralized configuration and directory setup."""

    def test_settings_attributes(self):
        """Verify that basic application metadata attributes are properly defined."""
        assert settings.APP_NAME == "ResumeMatch AI"
        assert settings.VERSION == "0.1.0"
        assert "CSE472" in settings.COURSE
        assert isinstance(settings.DEBUG, bool)
        assert settings.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def test_paths_exist_or_creatable(self):
        """Verify that directory paths resolve properly."""
        assert isinstance(settings.BASE_DIR, Path)
        assert (settings.BASE_DIR / "app.py").exists()

        assert isinstance(settings.DATA_DIR, Path)
        assert isinstance(settings.LOG_DIR, Path)
        assert isinstance(settings.SAMPLE_DATA_DIR, Path)
        assert isinstance(settings.MODEL_CACHE_DIR, Path)

        # Trigger ensure_directories and assert existence
        settings.ensure_directories()
        assert settings.DATA_DIR.exists()
        assert settings.LOG_DIR.exists()
        assert settings.SAMPLE_DATA_DIR.exists()
        assert settings.MODEL_CACHE_DIR.exists()

    def test_planned_features_structure(self):
        """Verify planned roadmap features list has all 7 days."""
        features = settings.PLANNED_FEATURES
        assert isinstance(features, list)
        assert len(features) == 7

        days = [f["day"] for f in features]
        for day_num in range(1, 8):
            assert f"Day {day_num}" in days

        # Verify Day 1 is marked Completed
        day_1 = next(f for f in features if f["day"] == "Day 1")
        assert day_1["status"] == "Completed"


class TestImports:
    """Smoke tests verifying that all modular packages and entrypoints import cleanly."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "config",
            "config.settings",
            "src",
            "src.parsers",
            "src.preprocessing",
            "src.nlp",
            "src.matching",
            "src.analytics",
            "src.utils",
            "src.utils.logger",
            "ui",
            "ui.landing",
            "app",
        ],
    )
    def test_module_import(self, module_name: str):
        """Ensure each module can be imported without syntax or import errors."""
        mod = importlib.import_module(module_name)
        assert mod is not None


class TestLogger:
    """Smoke tests for the logging system."""

    def test_get_logger(self):
        """Ensure get_logger returns an active Logger instance."""
        test_logger = get_logger("test_smoke")
        assert isinstance(test_logger, logging.Logger)
        assert test_logger.name == "test_smoke"

        # Verify logging does not raise exceptions
        test_logger.info("Smoke test log entry from pytest")
        test_logger.debug("Debug log entry")

    def test_file_handler_creation(self, tmp_path):
        """Verify that a file handler can write into the designated log file."""
        logger_name = "test_file_logger"
        logger = setup_logger(name=logger_name, log_file="test.log")
        assert any(
            isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers
        )


class TestUIModule:
    """Smoke tests for UI package exports."""

    def test_ui_callable(self):
        """Verify that render_landing_page is imported and callable."""
        from ui import render_landing_page

        assert callable(render_landing_page)
