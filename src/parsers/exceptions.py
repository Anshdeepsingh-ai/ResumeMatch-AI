"""
Exception classes for the ResumeMatch AI document parsing module.
"""


class ParserError(Exception):
    """Base exception for all document parsing errors."""
    pass


class FileNotFoundParserError(ParserError):
    """Raised when the specified document file does not exist."""
    pass


class UnsupportedFormatError(ParserError):
    """Raised when the document file format or extension is not supported."""
    pass


class FileTooLargeError(ParserError):
    """Raised when the file size exceeds the configured maximum allowable limit."""
    pass


class CorruptFileError(ParserError):
    """Raised when the document file is corrupted, malformed, or unreadable by the underlying engine."""
    pass


class EncodingError(ParserError):
    """Raised when a text document cannot be decoded using strict UTF-8 encoding."""
    pass


class EmptyDocumentError(ParserError):
    """
    Raised when the document is empty (0 bytes) or contains no extractable text.
    Note: For PDFs, this also occurs on image-only scanned documents where OCR is not enabled.
    """
    pass
