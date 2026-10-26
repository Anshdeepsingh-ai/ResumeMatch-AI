"""
Text Preprocessing and Normalization Module for ResumeMatch AI.
Course: CSE472 Deep Learning for NLP
Milestone: Day 3 Text Preprocessing & Normalization
"""

from src.preprocessing.normalizer import (
    clean_extraction_artifacts,
    normalize_line_endings,
    normalize_text,
    normalize_unicode,
    normalize_whitespace,
    remove_control_characters,
)

__all__ = [
    "normalize_text",
    "normalize_unicode",
    "normalize_line_endings",
    "normalize_whitespace",
    "remove_control_characters",
    "clean_extraction_artifacts",
]
