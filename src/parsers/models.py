"""
Data models for the ResumeMatch AI document parsing module.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ParsedDocument:
    """
    Structured representation of an extracted document.

    Attributes:
        text: Normalized textual content preserving paragraph and line boundaries.
        file_path: Absolute or resolved path to the source file.
        file_name: Base name of the file (including extension).
        file_type: Canonical format string ('pdf', 'docx', 'txt').
        file_size_bytes: Size of the original document file on disk.
        character_count: Total number of characters in the extracted text.
        word_count: Total number of words in the extracted text.
        page_count: Total number of pages (populated for PDF files, None otherwise).
        metadata: Dictionary containing format-specific or technical attributes.
    """

    text: str
    file_path: Path
    file_name: str
    file_type: str
    file_size_bytes: int
    character_count: int
    word_count: int
    page_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
