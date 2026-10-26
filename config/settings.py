"""
Centralized Configuration Module for ResumeMatch AI.
Course: CSE472 Deep Learning for NLP
Milestone: Day 1 Project Foundation
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# Attempt to load environment variables from .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env from project root
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    """Application-wide settings and environment configuration."""

    # Project Metadata
    APP_NAME: str = os.getenv("APP_NAME", "ResumeMatch AI")
    VERSION: str = "0.1.0"
    COURSE: str = "CSE472 Deep Learning for NLP"
    TIMEFRAME: str = "7-Day Project Plan"
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # Base Paths
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def DATA_DIR(self) -> Path:
        data_path = os.getenv("DATA_DIR", "data")
        path = Path(data_path)
        return path if path.is_absolute() else self.BASE_DIR / path

    @property
    def SAMPLE_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "sample"

    @property
    def LOG_DIR(self) -> Path:
        log_path = os.getenv("LOG_DIR", "logs")
        path = Path(log_path)
        return path if path.is_absolute() else self.BASE_DIR / path

    @property
    def MODEL_CACHE_DIR(self) -> Path:
        model_path = os.getenv("MODEL_CACHE_DIR", "models")
        path = Path(model_path)
        return path if path.is_absolute() else self.BASE_DIR / path

    # Server Settings
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    STREAMLIT_HOST: str = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")

    # Document Parsing Configuration (Day 2 Milestone)
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    SUPPORTED_EXTENSIONS: tuple = (".pdf", ".docx", ".txt")

    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    # Planned Roadmap Features (Milestones for Days 1 to 7)
    PLANNED_FEATURES: List[Dict[str, str]] = field(default_factory=lambda: [
        {
            "day": "Day 1",
            "name": "Project Foundation & Landing Page",
            "status": "Completed",
            "description": "Establish modular architecture, centralized settings, logging, smoke testing, and UI foundation."
        },
        {
            "day": "Day 2",
            "name": "Document Extraction (PDF & DOCX)",
            "status": "Completed",
            "description": "Robust extraction of raw text and metadata from PDF, DOCX, and TXT resume files."
        },
        {
            "day": "Day 3",
            "name": "Text Preprocessing & Cleaning",
            "status": "Planned",
            "description": "Tokenization, stopword normalization, section segmenting, and structured entity parsing."
        },
        {
            "day": "Day 4",
            "name": "Deep Learning NLP & Embeddings",
            "status": "Planned",
            "description": "Dense contextual representations using Sentence Transformers and deep contextual NLP models."
        },
        {
            "day": "Day 5",
            "name": "Semantic Matching & Skill Extraction",
            "status": "Planned",
            "description": "Cosine similarity scoring, skill taxonomy alignment, and weighted hybrid match formulas."
        },
        {
            "day": "Day 6",
            "name": "Candidate Ranking & Analytics",
            "status": "Planned",
            "description": "Top-K candidate sorting, skill gap visualization, analytics, and CSV report export."
        },
        {
            "day": "Day 7",
            "name": "End-to-End Integration & Polish",
            "status": "Planned",
            "description": "Interactive candidate dashboard, batch upload pipeline, comprehensive testing, and deployment."
        }
    ])

    def ensure_directories(self) -> None:
        """Ensure runtime directories (logs, data, models) exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Global singleton settings instance
settings = Settings()
settings.ensure_directories()
