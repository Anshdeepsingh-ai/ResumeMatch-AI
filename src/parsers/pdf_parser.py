"""
PDF Document Parser using PyMuPDF (fitz).
Course: CSE472 Deep Learning for NLP
Milestone: Day 2 Resume Parser Module
"""

from pathlib import Path
from typing import Dict, Tuple

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from src.parsers.exceptions import CorruptFileError, EmptyDocumentError


def extract_pdf_text(file_path: Path) -> Tuple[str, int, Dict[str, int]]:
    """
    Extract structured textual content from a PDF document using PyMuPDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Tuple containing:
            - text (str): Extracted text content preserving paragraph boundaries.
            - page_count (int): Total number of pages in the PDF.
            - metadata (Dict[str, int]): Technical metadata including page count.

    Raises:
        EmptyDocumentError: If the PDF contains 0 pages or contains no extractable text
                           (e.g., blank pages or scanned images without an embedded text layer).
        CorruptFileError: If the PDF is malformed, password-protected, or corrupted.
    """
    doc = None
    try:
        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            raise CorruptFileError(f"PDF file is corrupt or invalid ({file_path.name}): {exc}") from exc

        # Check for encrypted / password-protected documents
        if doc.is_encrypted and not doc.authenticate(""):
            raise CorruptFileError(f"PDF document is encrypted or password-protected: {file_path.name}")

        page_count = len(doc)
        if page_count == 0:
            raise EmptyDocumentError(f"PDF file contains 0 pages: {file_path.name}")

        page_contents = []
        for page_idx, page in enumerate(doc):
            try:
                page_text = page.get_text("text")
            except Exception as exc:
                raise CorruptFileError(f"Failed extracting text from page {page_idx + 1} of {file_path.name}: {exc}") from exc

            clean_page = page_text.strip()
            if clean_page:
                page_contents.append(clean_page)

        full_text = "\n\n".join(page_contents).strip()

        # Detect documents with no extractable text (e.g. scanned PDFs without OCR)
        if not full_text:
            raise EmptyDocumentError(
                f"Document contains no extractable text: {file_path.name}. "
                "Note: Scanned or image-only documents without an embedded text layer "
                "require OCR, which is not currently supported."
            )

        metadata = {
            "page_count": page_count,
            "non_empty_pages": len(page_contents),
        }

        return full_text, page_count, metadata

    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
