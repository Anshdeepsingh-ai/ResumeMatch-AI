"""
CSV export service for candidate ranking reports and batch evaluations.
Course: CSE472 Deep Learning for NLP
Milestone: Day 6 Candidate Ranking & Analytics
"""

import csv
import io
from typing import List, Optional

from src.analytics.models import CandidateResult

CSV_HEADERS: List[str] = [
    "Rank",
    "Candidate Identifier",
    "Filename",
    "Status",
    "Overall Score",
    "Similarity Score",
    "Skill Coverage Score",
    "Matching Method",
    "Fallback Active",
    "Matched Skills",
    "Missing Skills",
    "Error Message",
]


class CSVExportService:
    """
    Service generating standardized RFC 4180 compliant CSV exports for candidate ranking batches.
    Safely handles Unicode, commas, quotes, and newlines without exposing full resume text.
    """

    def __init__(self, headers: Optional[List[str]] = None) -> None:
        self.headers = headers or list(CSV_HEADERS)

    def export(self, candidates: List[CandidateResult]) -> str:
        """
        Format a list of CandidateResult instances into an RFC 4180 CSV string.

        Args:
            candidates: List of CandidateResult instances.

        Returns:
            Formatted CSV string.
        """
        output = io.StringIO()
        writer = csv.writer(
            output,
            dialect="excel",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\r\n",
        )

        # Write header row
        writer.writerow(self.headers)

        for c in candidates:
            rank_str = str(c.rank) if c.rank is not None else "N/A"
            overall_str = f"{c.overall_score:.2f}" if c.overall_score is not None else ""
            sim_str = f"{c.similarity_score:.2f}" if c.similarity_score is not None else ""
            skill_str = f"{c.skill_coverage_score:.2f}" if c.skill_coverage_score is not None else ""

            if c.is_fallback is True:
                fallback_str = "Yes"
            elif c.is_fallback is False:
                fallback_str = "No"
            else:
                fallback_str = ""

            matched_skills_str = "; ".join(c.matched_skills)
            missing_skills_str = "; ".join(c.missing_skills)

            row = [
                rank_str,
                c.candidate_id,
                c.filename,
                c.status,
                overall_str,
                sim_str,
                skill_str,
                c.method or "",
                fallback_str,
                matched_skills_str,
                missing_skills_str,
                c.error_message or "",
            ]
            writer.writerow(row)

        return output.getvalue()

    def export_bytes(
        self,
        candidates: List[CandidateResult],
        encoding: str = "utf-8-sig",
    ) -> bytes:
        """
        Format a list of CandidateResult instances into encoded CSV bytes.

        Args:
            candidates: List of CandidateResult instances.
            encoding: Target encoding (defaults to 'utf-8-sig' for cross-platform Excel compatibility).

        Returns:
            Encoded bytes suitable for HTTP download or file streaming.
        """
        csv_text = self.export(candidates)
        return csv_text.encode(encoding, errors="replace")


def export_candidates_to_csv(
    candidates: List[CandidateResult],
    headers: Optional[List[str]] = None,
) -> str:
    """
    Convenience function exporting candidates to an RFC 4180 CSV string.

    Args:
        candidates: List of CandidateResult objects.
        headers: Optional custom column headers.

    Returns:
        Formatted CSV text.
    """
    service = CSVExportService(headers=headers)
    return service.export(candidates)


def export_candidates_to_csv_bytes(
    candidates: List[CandidateResult],
    encoding: str = "utf-8-sig",
    headers: Optional[List[str]] = None,
) -> bytes:
    """
    Convenience function exporting candidates to downloadable CSV bytes.

    Args:
        candidates: List of CandidateResult objects.
        encoding: Target character encoding (default: 'utf-8-sig').
        headers: Optional custom column headers.

    Returns:
        Encoded CSV bytes.
    """
    service = CSVExportService(headers=headers)
    return service.export_bytes(candidates, encoding=encoding)
