"""
Text Preprocessing and Normalization Module for ResumeMatch AI.
Course: CSE472 Deep Learning for NLP
Milestone: Day 3 Text Preprocessing & Normalization
"""

import re
import unicodedata
from typing import Any

# Invisible zero-width and formatting characters common in extracted documents
_INVISIBLE_CHARS = (
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # zero-width no-break space / byte order mark
    "\u2060",  # word joiner
    "\u00ad",  # soft hyphen
)

# Common non-standard bullet characters used in resumes
_BULLET_PATTERN = re.compile(r"(?m)^(\s*)[●○▪■‣⁃▶►✔✓]\s*")

# Decorative horizontal divider lines (e.g., ---------------- or ================)
_DIVIDER_PATTERN = re.compile(r"(?m)^[-=_~*]{4,}\s*$")

# Wrapped word hyphenation at line breaks (e.g., develop-\nment -> development)
_HYPHENATION_PATTERN = re.compile(r"(\b[A-Za-z]{2,})-\n([a-z]{2,}\b)")

# Exotic quotation marks mapped to standard ASCII quotes
_SINGLE_QUOTE_PATTERN = re.compile(r"[\u2018\u2019\u201a\u201b\u02bb\u02bc`´]")
_DOUBLE_QUOTE_PATTERN = re.compile(r"[\u201c\u201d\u201e\u201f\u00ab\u00bb]")

# Exotic whitespace characters mapped to regular ASCII space
_EXOTIC_SPACES_PATTERN = re.compile(r"[\u00a0\u202f\u2007\u2009\u3000]")

# Dash variations mapped to standard hyphen or spaced hyphen
_SPACED_DASH_PATTERN = re.compile(r"\s*[–—‒―−]\s*")
_INLINE_DASH_PATTERN = re.compile(r"(?<=\w)[–—‒―−](?=\w)")


def _validate_text_input(text: Any) -> None:
    """Validate that input text is a string."""
    if not isinstance(text, str):
        raise TypeError(f"Input text must be a string, got {type(text).__name__}")


def normalize_line_endings(text: str) -> str:
    """
    Normalize various line endings and page-break characters into standard Unix newlines.

    Converts:
        - CRLF (\\r\\n) and CR (\\r) to LF (\\n).
        - Form feeds (\\f, often page breaks) to double newlines (\\n\\n).
        - Vertical tabs (\\v) and line separators (\\u2028) to single newlines.
        - Paragraph separators (\\u2029) to double newlines.

    Args:
        text: Input string.

    Returns:
        String with standardized line endings.

    Raises:
        TypeError: If text is not a string.
    """
    _validate_text_input(text)
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\f", "\n\n").replace("\v", "\n")
    normalized = normalized.replace("\u2028", "\n").replace("\u2029", "\n\n")
    return normalized


def remove_control_characters(text: str) -> str:
    """
    Remove non-printable control characters while preserving meaningful whitespace (\\n and \\t).

    Removes ASCII control characters (0x00-0x08, 0x0B, 0x0E-0x1F, 0x7F) and
    Unicode category 'Cc' characters, leaving text legible and clean for NLP.

    Args:
        text: Input string.

    Returns:
        String without unwanted control codes.

    Raises:
        TypeError: If text is not a string.
    """
    _validate_text_input(text)
    if not text:
        return ""

    return "".join(
        ch for ch in text
        if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text consistently using NFKC normalization.

    Performs:
        - Removal of zero-width and invisible format characters (BOM, ZWSP, ZWNJ, etc.).
        - Compatibility composition (NFKC), resolving typographic ligatures (fi, fl)
          and full-width characters while strictly preserving accented characters (é, ü, ñ).
        - Conversion of curly/smart quotes into standard ASCII quotes.
        - Conversion of non-breaking and thin spaces into standard spaces.
        - Normalization of em-dashes and en-dashes to standard hyphens.

    Args:
        text: Input string.

    Returns:
        Unicode-normalized string.

    Raises:
        TypeError: If text is not a string.
    """
    _validate_text_input(text)
    if not text:
        return ""

    # Remove invisible formatting characters
    cleaned = text
    for char in _INVISIBLE_CHARS:
        cleaned = cleaned.replace(char, "")

    # Apply NFKC decomposition and recomposition
    cleaned = unicodedata.normalize("NFKC", cleaned)

    # Standardize quotation marks
    cleaned = _SINGLE_QUOTE_PATTERN.sub("'", cleaned)
    cleaned = _DOUBLE_QUOTE_PATTERN.sub('"', cleaned)

    # Standardize spaces
    cleaned = _EXOTIC_SPACES_PATTERN.sub(" ", cleaned)

    # Standardize dashes (inline between words first, then spaced separators)
    cleaned = _INLINE_DASH_PATTERN.sub("-", cleaned)
    cleaned = _SPACED_DASH_PATTERN.sub(" - ", cleaned)

    return cleaned


def clean_extraction_artifacts(text: str) -> str:
    """
    Handle common document extraction artifacts from PDF and DOCX files.

    Performs:
        - Repairing wrapped word hyphenation across line breaks (e.g., develop-\\nment -> development).
        - Removing decorative horizontal divider lines (e.g., repeating dashes or underscores).
        - Standardizing diverse bullet symbols (●, ▪, ►, etc.) to standard bullet markers (• ).
        - Ensuring consistent spacing following bullet markers.

    Args:
        text: Input string.

    Returns:
        String with extraction artifacts resolved.

    Raises:
        TypeError: If text is not a string.
    """
    _validate_text_input(text)
    if not text:
        return ""

    # Fix word breaks across lines (e.g., develop-\nment -> development)
    cleaned = _HYPHENATION_PATTERN.sub(r"\1\2", text)

    # Remove decorative divider lines (e.g., ---------------- or ================)
    cleaned = _DIVIDER_PATTERN.sub("", cleaned)

    # Standardize exotic bullet points to standard bullet symbol
    cleaned = _BULLET_PATTERN.sub(r"\1• ", cleaned)

    # Ensure space after bullet point if directly attached to text
    cleaned = re.sub(r"(?m)^(\s*•)([^\s•])", r"\1 \2", cleaned)

    return cleaned


def normalize_whitespace(text: str) -> str:
    """
    Normalize horizontal and vertical whitespace.

    Performs:
        - Collapsing multiple consecutive spaces and tabs into a single space within each line.
        - Stripping leading and trailing whitespace from each line.
        - Collapsing excessive consecutive blank lines into at most one blank line (\\n\\n).
        - Stripping leading and trailing whitespace from the overall document.

    Args:
        text: Input string.

    Returns:
        String with normalized whitespace and paragraph separation.

    Raises:
        TypeError: If text is not a string.
    """
    _validate_text_input(text)
    if not text:
        return ""

    # Normalize each line individually
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)

    # Collapse 3 or more consecutive newlines into 2 (preserving standard paragraph gaps)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def normalize_text(text: str) -> str:
    """
    Master preprocessing pipeline for normalizing extracted resume and job description text.

    Executes all normalization stages in a deterministic sequence:
        1. Line ending normalization (standardizing to \\n, converting page breaks).
        2. Removal of non-printable control characters.
        3. Unicode NFKC normalization, quote standardizing, dash mapping.
        4. Cleaning of extraction artifacts (word de-hyphenation, bullet points, divider lines).
        5. Whitespace and blank line normalization.

    Guarantees:
        - Preserves technical punctuation and terms: C++, C#, .NET, Node.js, version numbers.
        - Preserves case, names, dates, contact information, and bulleted structure.
        - Does NOT apply lossy NLP transforms: no lowercasing, no stopword removal, no stemming.
        - Deterministic and idempotent: normalize_text(normalize_text(text)) == normalize_text(text).
        - Returns empty string for empty or whitespace-only inputs.

    Args:
        text: Raw extracted document text.

    Returns:
        Clean, normalized text suitable for downstream NLP embedding and matching.

    Raises:
        TypeError: If text is not a string.
    """
    _validate_text_input(text)
    if not text or text.isspace():
        return ""

    pipeline = normalize_line_endings(text)
    pipeline = remove_control_characters(pipeline)
    pipeline = normalize_unicode(pipeline)
    pipeline = clean_extraction_artifacts(pipeline)
    pipeline = normalize_whitespace(pipeline)

    return pipeline
