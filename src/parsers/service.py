"""
Document Parser Service for ResumeMatch AI.
Coordinates format validation, file safety checks, and format-specific extraction.
Course: CSE472 Deep Learning for NLP
Milestone: Day 2 Resume Parser Module
"""

from pathlib import Path
from typing import Optional, Union

from config.settings import settings
from src.parsers.docx_parser import extract_docx_text
from src.parsers.exceptions import (
    EmptyDocumentError,
    FileNotFoundParserError,
    FileTooLargeError,
    ParserError,
    UnsupportedFormatError,
)
from src.parsers.models import ParsedDocument
from src.parsers.pdf_parser import extract_pdf_text
from src.parsers.txt_parser import extract_txt_text
from src.utils.logger import get_logger

logger = get_logger("parsers")


class DocumentParser:
    """
    Service responsible for parsing and extracting clean textual content from
    PDF, DOCX, and TXT documents.
    """

    def __init__(self, max_file_size_bytes: Optional[int] = None):
        """
        Initialize DocumentParser.

        Args:
            max_file_size_bytes: Maximum allowed file size in bytes.
                                 Defaults to settings.MAX_FILE_SIZE_BYTES.
        """
        self.max_file_size_bytes = (
            max_file_size_bytes
            if max_file_size_bytes is not None
            else settings.MAX_FILE_SIZE_BYTES
        )
        self.supported_extensions = set(ext.lower() for ext in settings.SUPPORTED_EXTENSIONS)

    def parse(self, file_path: Union[str, Path]) -> ParsedDocument:
        """
        Parse a document file and return structured document representation.

        Args:
            file_path: Path to the target document (PDF, DOCX, or TXT).

        Returns:
            ParsedDocument containing extracted text and non-PII technical metadata.

        Raises:
            FileNotFoundParserError: If the file does not exist.
            UnsupportedFormatError: If the file extension is not in supported_extensions.
            FileTooLargeError: If file size exceeds max_file_size_bytes.
            EmptyDocumentError: If the file is 0 bytes or yields no extractable text.
            CorruptFileError: If the file is corrupted or unreadable.
            EncodingError: If a text file contains invalid UTF-8 bytes.
            ParserError: If general permission or I/O failure occurs.
        """
        path = Path(file_path).resolve()

        # 1. Verify file existence
        if not path.is_file():
            logger.warning("Document extraction failed: file not found (%s)", path.name)
            raise FileNotFoundParserError(f"Document file not found: {path}")

        # 2. Verify file extension
        suffix = path.suffix.lower()
        if suffix not in self.supported_extensions:
            supported_str = ", ".join(sorted(self.supported_extensions))
            logger.warning("Unsupported file extension '%s' for %s", suffix, path.name)
            raise UnsupportedFormatError(
                f"Unsupported file extension '{suffix}'. Supported formats: {supported_str}"
            )

        # 3. Check file size
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            raise ParserError(f"Unable to read file metadata for {path.name}: {exc}") from exc

        if file_size == 0:
            logger.warning("Document is empty (0 bytes): %s", path.name)
            raise EmptyDocumentError(f"Document file is empty (0 bytes): {path.name}")

        if file_size > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            logger.warning(
                "Document %s exceeds size limit (%.2f MB > %.2f MB)",
                path.name,
                actual_mb,
                max_mb,
            )
            raise FileTooLargeError(
                f"File size ({actual_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.2f} MB."
            )

        # Log extraction start (safe metadata only - no PII or resume content)
        logger.info("Extracting text from '%s' (type=%s, size=%d bytes)", path.name, suffix, file_size)

        # 4. Route to format-specific extractor
        file_type = suffix.lstrip(".")
        page_count: Optional[int] = None
        metadata = {}

        if suffix == ".pdf":
            text, page_count, metadata = extract_pdf_text(path)
        elif suffix == ".docx":
            text, metadata = extract_docx_text(path)
        elif suffix == ".txt":
            text, metadata = extract_txt_text(path)
        else:
            raise UnsupportedFormatError(f"No parser implemented for '{suffix}'")

        # 5. Build parsed document model
        char_count = len(text)
        word_count = len(text.split())

        logger.info(
            "Successfully extracted '%s': %d characters, %d words (type=%s)",
            path.name,
            char_count,
            word_count,
            file_type,
        )

        return ParsedDocument(
            text=text,
            file_path=path,
            file_name=path.name,
            file_type=file_type,
            file_size_bytes=file_size,
            character_count=char_count,
            word_count=word_count,
            page_count=page_count,
            metadata=metadata,
        )


def parse_document(
    file_path: Union[str, Path],
    max_file_size_bytes: Optional[int] = None,
) -> ParsedDocument:
    """
    Convenience function to parse a document using DocumentParser.

    Args:
        file_path: Path to the target document.
        max_file_size_bytes: Optional custom maximum size limit in bytes.

    Returns:
        ParsedDocument: Extracted document data and metadata.
    """
    parser = DocumentParser(max_file_size_bytes=max_file_size_bytes)
    return parser.parse(file_path)
