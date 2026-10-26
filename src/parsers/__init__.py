"""
Document parsing and text extraction module for ResumeMatch AI.
Course: CSE472 Deep Learning for NLP
Milestone: Day 2 Resume Parser Module
"""

from src.parsers.docx_parser import extract_docx_text
from src.parsers.exceptions import (
    CorruptFileError,
    EmptyDocumentError,
    EncodingError,
    FileNotFoundParserError,
    FileTooLargeError,
    ParserError,
    UnsupportedFormatError,
)
from src.parsers.models import ParsedDocument
from src.parsers.pdf_parser import extract_pdf_text
from src.parsers.service import DocumentParser, parse_document
from src.parsers.txt_parser import extract_txt_text

__all__ = [
    "DocumentParser",
    "ParsedDocument",
    "parse_document",
    "extract_pdf_text",
    "extract_docx_text",
    "extract_txt_text",
    "ParserError",
    "FileNotFoundParserError",
    "UnsupportedFormatError",
    "FileTooLargeError",
    "CorruptFileError",
    "EncodingError",
    "EmptyDocumentError",
]
