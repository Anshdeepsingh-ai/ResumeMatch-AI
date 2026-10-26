"""
Comprehensive unit tests for the ResumeMatch AI text preprocessing and normalization module.
Course: CSE472 Deep Learning for NLP
Milestone: Day 3 Text Preprocessing & Normalization
"""

import pytest
from src.preprocessing import (
    clean_extraction_artifacts,
    normalize_line_endings,
    normalize_text,
    normalize_unicode,
    normalize_whitespace,
    remove_control_characters,
)


# ==============================================================================
# 1. Normal Text and Whitespace Normalization Tests
# ==============================================================================


class TestWhitespaceNormalization:
    """Tests verifying whitespace normalization, collapsing, and trimming."""

    def test_repeated_inline_spaces(self):
        """Verify multiple consecutive spaces within a line collapse to a single space."""
        raw = "Senior   Machine    Learning   Engineer"
        assert normalize_whitespace(raw) == "Senior Machine Learning Engineer"

    def test_leading_and_trailing_whitespace(self):
        """Verify leading and trailing whitespace on lines and document are stripped."""
        raw = "   John Doe   \n   Software Engineer   \n   "
        expected = "John Doe\nSoftware Engineer"
        assert normalize_whitespace(raw) == expected

    def test_tabs_within_lines(self):
        """Verify tabs within lines are normalized to single spaces."""
        raw = "Skills:\tPython\t\tPyTorch\tDocker"
        assert normalize_whitespace(raw) == "Skills: Python PyTorch Docker"

    def test_excessive_blank_lines(self):
        """Verify multiple consecutive blank lines collapse to a single blank line (\\n\\n)."""
        raw = "Section 1\n\n\n\n\n\nSection 2\n\n\nSection 3"
        expected = "Section 1\n\nSection 2\n\nSection 3"
        assert normalize_whitespace(raw) == expected


# ==============================================================================
# 2. Line Ending Normalization Tests
# ==============================================================================


class TestLineEndingNormalization:
    """Tests verifying line ending conversion across Windows, Unix, and Mac formats."""

    def test_windows_crlf(self):
        """Verify Windows CRLF (\\r\\n) is converted to Unix LF (\\n)."""
        raw = "Line 1\r\nLine 2\r\nLine 3"
        assert normalize_line_endings(raw) == "Line 1\nLine 2\nLine 3"

    def test_legacy_mac_cr(self):
        """Verify legacy Mac CR (\\r) is converted to LF (\\n)."""
        raw = "Line 1\rLine 2\rLine 3"
        assert normalize_line_endings(raw) == "Line 1\nLine 2\nLine 3"

    def test_mixed_line_endings(self):
        """Verify mixed line endings in a single document are normalized."""
        raw = "Line 1\r\nLine 2\nLine 3\rLine 4"
        assert normalize_line_endings(raw) == "Line 1\nLine 2\nLine 3\nLine 4"

    def test_form_feed_page_breaks(self):
        """Verify form feed characters (\\f) are converted to double newlines."""
        raw = "Page 1 Content\fPage 2 Content"
        assert normalize_line_endings(raw) == "Page 1 Content\n\nPage 2 Content"

    def test_vertical_tab_and_separators(self):
        """Verify vertical tabs (\\v) and Unicode line/para separators are normalized."""
        raw = "Item 1\vItem 2\u2028Item 3\u2029Paragraph 2"
        expected = "Item 1\nItem 2\nItem 3\n\nParagraph 2"
        assert normalize_line_endings(raw) == expected


# ==============================================================================
# 3. Unicode and Accented Characters Tests
# ==============================================================================


class TestUnicodeNormalization:
    """Tests verifying Unicode NFKC normalization, quote standardizing, and dash handling."""

    def test_accented_characters_preserved(self):
        """Verify accented international characters are strictly preserved."""
        raw = "François Müller, Ph.D. - Zürich résumé & café"
        normalized = normalize_unicode(raw)
        assert "François" in normalized
        assert "Müller" in normalized
        assert "Zürich" in normalized
        assert "résumé" in normalized
        assert "café" in normalized

    def test_decomposed_accents_recomposed(self):
        """Verify decomposed combining characters (e + combining acute) become precomposed."""
        decomposed = "re\u0301sume\u0301"  # e + combining acute
        normalized = normalize_unicode(decomposed)
        assert normalized == "résumé"

    def test_typographic_ligatures(self):
        """Verify PDF ligatures (fi, fl, ffi) decompose to individual letters under NFKC."""
        raw = "\ufb01le \ufb02ow \ufb03cient"
        assert normalize_unicode(raw) == "file flow fficient"

    def test_curly_quotes_standardized(self):
        """Verify smart/curly quotes become standard ASCII single and double quotes."""
        raw = "“Smart Double Quotes” and ‘Smart Single Quotes’"
        expected = '"Smart Double Quotes" and \'Smart Single Quotes\''
        assert normalize_unicode(raw) == expected

    def test_invisible_and_zero_width_chars(self):
        """Verify zero-width spaces, BOM, and soft hyphens are removed."""
        raw = "N\u200bo\u200cd\u200de\ufeff.\u2060js\u00ad"
        assert normalize_unicode(raw) == "Node.js"

    def test_dashes_standardized(self):
        """Verify em-dashes and en-dashes are standardized to hyphens."""
        raw_range = "2020 – 2024"
        raw_inline = "2020–2024"
        raw_em = "Experience — Senior Role"
        assert normalize_unicode(raw_range) == "2020 - 2024"
        assert normalize_unicode(raw_inline) == "2020-2024"
        assert normalize_unicode(raw_em) == "Experience - Senior Role"

    def test_exotic_whitespace_spaces(self):
        """Verify non-breaking spaces and thin spaces map to regular ASCII space."""
        raw = "Python\u00a0Developer\u202fin\u2009AI"
        assert normalize_unicode(raw) == "Python Developer in AI"


# ==============================================================================
# 4. Control Characters Tests
# ==============================================================================


class TestControlCharacters:
    """Tests verifying removal of non-printable control codes while keeping newlines and tabs."""

    def test_strip_null_and_bell_bytes(self):
        """Verify NUL, BEL, backspace, and escape sequences are eliminated."""
        raw = "Clean\x00Text\x07With\x08Control\x1b[31mChars\x7f"
        assert remove_control_characters(raw) == "CleanTextWithControl[31mChars"

    def test_preserve_newline_and_tab(self):
        """Verify newlines and tabs are not stripped by remove_control_characters."""
        raw = "Line 1\n\tTabbed Indent\nLine 2"
        assert remove_control_characters(raw) == raw


# ==============================================================================
# 5. Extraction Artifacts Cleaning Tests
# ==============================================================================


class TestExtractionArtifacts:
    """Tests verifying cleanup of line-wrap hyphens, dividers, and bullet standardization."""

    def test_hyphenated_line_break_repair(self):
        """Verify words split across lines by hyphens are properly joined."""
        raw = "Specializing in the develop-\nment of neural archi-\ntecture."
        expected = "Specializing in the development of neural architecture."
        assert clean_extraction_artifacts(raw) == expected

    def test_decorative_divider_lines_removed(self):
        """Verify lines of repeating hyphens, equals, or underscores are removed."""
        raw = "EXPERIENCE\n--------------------\n• Role 1\n====================\n• Role 2"
        cleaned = clean_extraction_artifacts(raw)
        assert "--------------------" not in cleaned
        assert "====================" not in cleaned
        assert "EXPERIENCE" in cleaned

    def test_exotic_bullet_symbols_standardized(self):
        """Verify non-standard bullet characters (●, ▪, ►, etc.) are converted to standard •."""
        raw = (
            "● Led machine learning team\n"
            "▪ Architected cloud backend\n"
            "► Optimized neural models\n"
            "✔ Delivered production systems\n"
            "•Existing standard bullet"
        )
        cleaned = clean_extraction_artifacts(raw)
        lines = cleaned.splitlines()
        for line in lines:
            assert line.startswith("• "), f"Line failed bullet check: {line}"


# ==============================================================================
# 6. Preservation of Critical Technical Terms & Resume Info
# ==============================================================================


class TestPreservationOfTechnicalTerms:
    """Tests verifying critical programming languages, frameworks, punctuation, and casing."""

    def test_preserve_cpp_csharp_dotnet_nodejs(self):
        """Verify C++, C#, .NET, and Node.js punctuation is not corrupted or stripped."""
        raw = "Expertise in C++, C#, .NET Framework, ASP.NET Core, and Node.js."
        cleaned = normalize_text(raw)
        assert "C++" in cleaned
        assert "C#" in cleaned
        assert ".NET Framework" in cleaned
        assert "ASP.NET Core" in cleaned
        assert "Node.js" in cleaned

    def test_preserve_version_numbers(self):
        """Verify version numbers and technical acronyms retain exact punctuation."""
        raw = "Python 3.12.14, PyTorch v2.1.0, CUDA 12.2, CI/CD, TCP/IP."
        cleaned = normalize_text(raw)
        assert "Python 3.12.14" in cleaned
        assert "PyTorch v2.1.0" in cleaned
        assert "CUDA 12.2" in cleaned
        assert "CI/CD" in cleaned
        assert "TCP/IP" in cleaned

    def test_preserve_case_and_acronyms(self):
        """Verify normalization does NOT lowercase text automatically."""
        raw = "Senior NLP Engineer at AWS & GCP working on LLMs and RAG."
        cleaned = normalize_text(raw)
        assert "Senior NLP Engineer" in cleaned
        assert "AWS" in cleaned
        assert "GCP" in cleaned
        assert "LLMs" in cleaned
        assert "RAG" in cleaned

    def test_preserve_contact_info_and_dates(self):
        """Verify email addresses, phone formats, URLs, and date ranges remain intact."""
        raw = (
            "synthetic.candidate@example.org | +1 (555) 019-2834 | github.com/synthetic\n"
            "Jan 2020 – Present (4 years)"
        )
        cleaned = normalize_text(raw)
        assert "synthetic.candidate@example.org" in cleaned
        assert "+1 (555) 019-2834" in cleaned
        assert "github.com/synthetic" in cleaned
        assert "Jan 2020 - Present" in cleaned


# ==============================================================================
# 7. Empty, Whitespace-Only, and Type Validation Tests
# ==============================================================================


class TestEdgeCasesAndValidation:
    """Tests verifying behavior on empty strings, whitespace, and invalid types."""

    @pytest.mark.parametrize("empty_input", ["", "   ", "\t", "\n\n", "  \r\n\t  \n  "])
    def test_normalize_text_empty_and_whitespace(self, empty_input):
        """Verify empty and whitespace-only strings return empty string predictably."""
        assert normalize_text(empty_input) == ""

    @pytest.mark.parametrize("empty_input", ["", "   ", "\t", "\n"])
    def test_sub_functions_empty_input(self, empty_input):
        """Verify sub-functions handle empty strings without crashing."""
        assert normalize_line_endings("") == ""
        assert remove_control_characters("") == ""
        assert normalize_unicode("") == ""
        assert clean_extraction_artifacts("") == ""
        assert normalize_whitespace("") == ""

    @pytest.mark.parametrize("invalid_input", [None, 123, 45.6, ["string"], {"text": "val"}])
    def test_type_validation_raises_type_error(self, invalid_input):
        """Verify passing non-string types raises TypeError explicitly."""
        with pytest.raises(TypeError, match="Input text must be a string"):
            normalize_text(invalid_input)

        with pytest.raises(TypeError, match="Input text must be a string"):
            normalize_line_endings(invalid_input)

        with pytest.raises(TypeError, match="Input text must be a string"):
            remove_control_characters(invalid_input)

        with pytest.raises(TypeError, match="Input text must be a string"):
            normalize_unicode(invalid_input)

        with pytest.raises(TypeError, match="Input text must be a string"):
            clean_extraction_artifacts(invalid_input)

        with pytest.raises(TypeError, match="Input text must be a string"):
            normalize_whitespace(invalid_input)


# ==============================================================================
# 8. Idempotency Tests
# ==============================================================================


class TestIdempotency:
    """Tests verifying that processing already-normalized text is strictly idempotent."""

    @pytest.mark.parametrize(
        "sample",
        [
            "Simple single line sentence.",
            "Paragraph 1\n\nParagraph 2\n\nParagraph 3",
            "• Bullet 1 with C++\n• Bullet 2 with Node.js and .NET\n• Bullet 3 with Python 3.12",
            "Dr. Jane Doe, Ph.D.\nZürich Institute of Technology (2018 - 2022)\nRole: Staff Scientist",
            "EXPERIENCE\n\n• Developed microservices using PyTorch and FastAPI.\n• Deployed on AWS ECS.",
        ],
    )
    def test_idempotent_normalize_text(self, sample):
        """Verify normalize_text(normalize_text(x)) == normalize_text(x)."""
        first_pass = normalize_text(sample)
        second_pass = normalize_text(first_pass)
        assert first_pass == second_pass

    def test_full_complex_synthetic_resume(self):
        """End-to-end synthetic resume preprocessing test verifying all stages together."""
        raw_resume = (
            "   \r\n\r\n"
            "Alex Mercer, M.Sc.\r\n"
            "alex.mercer@synthetic-mail.org | +1 (555) 012-3456\r\n"
            "----------------------------------------\r\n"
            "SUMMARY\r\n"
            "Experienced NLP Researcher with 6+ years building deep learn-\r\n"
            "ing systems in C++, Python 3.12, and Node.js.\r\n"
            "\x00\x07\f"  # null byte, bell byte, form-feed page break
            "TECHNICAL SKILLS\r\n"
            "========================================\r\n"
            "● Languages: Python, C++, C#, SQL, JavaScript\r\n"
            "▪ Frameworks: PyTorch, Hugging Face, .NET Core, FastAPI\r\n"
            "► Cloud: AWS (SageMaker, ECS), Docker, CI/CD\r\n"
            "\r\n\r\n\r\n"
            "EXPERIENCE\r\n"
            "Senior Machine Learning Engineer — TechCorp (2020 – Present)\r\n"
            "•Designed and fine-tuned BERT and Sentence-BERT models.\r\n"
            "• Improved semantic search ranking by 24% across 1M+ queries.\r\n"
            "   \r\n"
        )

        cleaned = normalize_text(raw_resume)

        # Assertions on cleaned output
        assert "Alex Mercer, M.Sc." in cleaned
        assert "----------------------------------------" not in cleaned
        assert "========================================" not in cleaned
        assert "deep learning systems" in cleaned  # de-hyphenated
        assert "C++, Python 3.12, and Node.js" in cleaned
        assert "\x00" not in cleaned
        assert "\x07" not in cleaned
        assert "• Languages: Python, C++, C#, SQL, JavaScript" in cleaned
        assert "• Frameworks: PyTorch, Hugging Face, .NET Core, FastAPI" in cleaned
        assert "• Cloud: AWS (SageMaker, ECS), Docker, CI/CD" in cleaned
        assert "TechCorp (2020 - Present)" in cleaned  # dash normalized
        assert "• Designed and fine-tuned" in cleaned  # space after bullet
        assert "\n\n\n" not in cleaned  # no excessive blank lines

        # Idempotency check
        assert normalize_text(cleaned) == cleaned
