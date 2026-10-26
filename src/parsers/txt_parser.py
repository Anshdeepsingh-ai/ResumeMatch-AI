"""
Plain Text Document Parser using strict UTF-8 decoding.
Course: CSE472 Deep Learning for NLP
Milestone: Day 2 Resume Parser Module
"""

from pathlib import Path
from typing import Dict, Tuple

from src.parsers.exceptions import EmptyDocumentError, EncodingError, ParserError


def extract_txt_text(file_path: Path) -> Tuple[str, Dict[str, int]]:
    """
    Extract textual content from a plain text file strictly decoded as UTF-8.

    Args:
        file_path: Path to the text file.

    Returns:
        Tuple containing:
            - text (str): Extracted text content preserving paragraph boundaries.
            - metadata (Dict[str, int]): Technical metadata including line count.

    Raises:
        EncodingError: If the document is not valid UTF-8.
        EmptyDocumentError: If the document contains no extractable text or only whitespace.
        ParserError: If an I/O error occurs while reading the file.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="strict") as file:
            content = file.read()
    except UnicodeDecodeError as exc:
        raise EncodingError(
            f"Failed to decode text file ({file_path.name}) as UTF-8: {exc}"
        ) from exc
    except OSError as exc:
        raise ParserError(f"I/O error reading text file ({file_path.name}): {exc}") from exc

    clean_text = content.strip()
    if not clean_text:
        raise EmptyDocumentError(f"Text document contains no extractable text: {file_path.name}")

    line_count = len([line for line in content.splitlines() if line.strip()])

    metadata = {
        "line_count": line_count,
        "raw_character_count": len(content),
    }

    return clean_text, metadata
