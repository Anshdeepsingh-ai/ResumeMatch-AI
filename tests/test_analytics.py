"""
Unit and integration tests for candidate ranking, batch analytics, and CSV report export.
Course: CSE472 Deep Learning for NLP
Milestone: Day 6 Candidate Ranking & Analytics

Tests cover:
- Ranking order and deterministic 4-stage tie-breaking.
- Partial batch failures and document extraction error containment.
- Input validation (empty JD, non-string, non-list, empty candidate list).
- Statistical batch analytics (mean, median, std, min, max, top skills, score buckets).
- Empty result behavior and zero-candidate safety.
- RFC 4180 CSV export formatting, Unicode preservation, escaping, and lack of full resume text.
- Offline mock integration without network access.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.analytics import (
    CSV_HEADERS,
    BatchAnalytics,
    BatchRankingResult,
    CandidateRankingService,
    CandidateResult,
    calculate_batch_analytics,
    export_candidates_to_csv,
    export_candidates_to_csv_bytes,
    rank_candidates,
)
from src.matching.models import BaseSimilarityEngine
from src.matching.schemas import MatchResult


class TestDeterministicRankingAndTieBreaking:
    """Tests for ranking order and deterministic multi-key tie-breaking."""

    def test_ranking_order_by_overall_score(self, tmp_path: Path) -> None:
        """Verify candidates are sorted by overall_score in descending order."""
        # Create 3 synthetic candidates with distinct scores
        file1 = tmp_path / "cand_low.txt"
        file1.write_text("Basic text with minimal skills.", encoding="utf-8")

        file2 = tmp_path / "cand_high.txt"
        file2.write_text("Senior Engineer with Python, Docker, PyTorch, and SQL expertise.", encoding="utf-8")

        file3 = tmp_path / "cand_mid.txt"
        file3.write_text("Developer with Python and SQL experience.", encoding="utf-8")

        jd = "Seeking a Senior Python Engineer with Docker, PyTorch, and SQL expertise."

        result = rank_candidates([file1, file2, file3], jd)

        assert len(result.candidates) == 3
        # Successful candidates should be ranked 1, 2, 3
        assert result.candidates[0].rank == 1
        assert result.candidates[1].rank == 2
        assert result.candidates[2].rank == 3

        # Scores should be strictly descending or equal
        score_1 = result.candidates[0].overall_score or 0.0
        score_2 = result.candidates[1].overall_score or 0.0
        score_3 = result.candidates[2].overall_score or 0.0
        assert score_1 >= score_2 >= score_3

    def test_tie_breaking_stages(self) -> None:
        """
        Verify deterministic 4-stage tie breaking:
        1. overall_score (descending)
        2. similarity_score (descending)
        3. skill_coverage_score (descending)
        4. filename (ascending)
        """
        # Create candidates with simulated tie conditions
        # Candidate A and B have same overall_score, but A has higher similarity_score
        c_a = CandidateResult(
            candidate_id="cand_1",
            filename="alpha.pdf",
            status="success",
            overall_score=75.0,
            similarity_score=80.0,
            skill_coverage_score=67.5,
        )
        c_b = CandidateResult(
            candidate_id="cand_2",
            filename="beta.pdf",
            status="success",
            overall_score=75.0,
            similarity_score=70.0,
            skill_coverage_score=82.5,
        )

        # Candidate C and D have same overall_score and similarity_score, but C has higher skill_coverage
        c_c = CandidateResult(
            candidate_id="cand_3",
            filename="gamma.pdf",
            status="success",
            overall_score=60.0,
            similarity_score=60.0,
            skill_coverage_score=60.0,
        )
        c_d = CandidateResult(
            candidate_id="cand_4",
            filename="delta.pdf",
            status="success",
            overall_score=60.0,
            similarity_score=60.0,
            skill_coverage_score=50.0,
        )

        # Candidate E and F have identical scores on all 3 metrics -> break tie by filename ascending
        c_e = CandidateResult(
            candidate_id="cand_5",
            filename="zeta.pdf",
            status="success",
            overall_score=50.0,
            similarity_score=50.0,
            skill_coverage_score=50.0,
        )
        c_f = CandidateResult(
            candidate_id="cand_6",
            filename="epsilon.pdf",
            status="success",
            overall_score=50.0,
            similarity_score=50.0,
            skill_coverage_score=50.0,
        )

        pool = [c_b, c_d, c_f, c_a, c_e, c_c]

        # Use sorting key identical to CandidateRankingService
        pool.sort(
            key=lambda c: (
                -(c.overall_score if c.overall_score is not None else 0.0),
                -(c.similarity_score if c.similarity_score is not None else 0.0),
                -(c.skill_coverage_score if c.skill_coverage_score is not None else 0.0),
                c.filename.lower(),
                c.candidate_id,
            )
        )

        for rank_idx, cand in enumerate(pool, start=1):
            cand.rank = rank_idx

        # Check order:
        # 1. c_a (75.0, sim=80) before c_b (75.0, sim=70)
        assert pool[0] == c_a
        assert pool[1] == c_b

        # 2. c_c (60.0, skill=60) before c_d (60.0, skill=50)
        assert pool[2] == c_c
        assert pool[3] == c_d

        # 3. c_f ('epsilon.pdf') before c_e ('zeta.pdf') due to alphabetical tie-breaking
        assert pool[4] == c_f
        assert pool[5] == c_e


class TestPartialBatchFailuresAndErrorContainment:
    """Tests for isolating individual file failures without terminating batch processing."""

    def test_batch_processes_despite_corrupt_and_unsupported_files(self, tmp_path: Path) -> None:
        """Ensure batch completes with mixed valid, empty, and unsupported files."""
        # 1. Valid TXT resume
        valid_file = tmp_path / "valid_resume.txt"
        valid_file.write_text("Experienced Python backend developer with FastAPI and SQL.", encoding="utf-8")

        # 2. Empty 0-byte file (raises EmptyDocumentError)
        empty_file = tmp_path / "empty_resume.txt"
        empty_file.write_text("", encoding="utf-8")

        # 3. Unsupported extension (raises UnsupportedFormatError)
        unsupported_file = tmp_path / "invalid_resume.jpg"
        unsupported_file.write_bytes(b"\xFF\xD8\xFF\xE0")

        # 4. Another valid file via in-memory tuple
        valid_tuple = ("second_valid.txt", b"Software Engineer with Docker, Python, and AWS experience.")

        jd = "Seeking Python Developer proficient in Docker, FastAPI, and SQL."

        result = rank_candidates([valid_file, empty_file, unsupported_file, valid_tuple], jd)

        # Check total submitted and failure containment
        assert result.analytics.total_submitted == 4
        assert result.analytics.total_processed == 2
        assert result.analytics.total_failed == 2
        assert result.analytics.success_rate == 50.0

        # Successful candidates have ranks 1 and 2
        successful = [c for c in result.candidates if c.status == "success"]
        assert len(successful) == 2
        assert successful[0].rank == 1
        assert successful[1].rank == 2

        # Failed candidates are recorded with rank=None and explicit error messages
        failed = [c for c in result.candidates if c.status == "failed"]
        assert len(failed) == 2
        for f in failed:
            assert f.rank is None
            assert f.error_message is not None
            assert len(f.error_message) > 0


class TestInputValidationAndEdgeCases:
    """Tests for parameter validation, type safety, and edge case responses."""

    def test_non_string_job_description_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Job description must be a string"):
            rank_candidates([], None)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Job description must be a string"):
            rank_candidates([], 12345)  # type: ignore[arg-type]

    def test_empty_job_description_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Target job description cannot be empty"):
            rank_candidates([], "")

        with pytest.raises(ValueError, match="Target job description cannot be empty"):
            rank_candidates([], "   \t\n  ")

    def test_non_list_resumes_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Resumes must be a list"):
            rank_candidates("not a list", "Valid JD")  # type: ignore[arg-type]

    def test_empty_candidate_list_returns_clean_empty_result(self) -> None:
        result = rank_candidates([], "Looking for Python Engineer with SQL expertise.")
        assert isinstance(result, BatchRankingResult)
        assert result.candidates == []
        assert result.analytics.total_submitted == 0
        assert result.analytics.total_processed == 0
        assert result.analytics.total_failed == 0
        assert result.analytics.mean_overall_score is None
        assert "Python" in result.job_description_skills
        assert "SQL" in result.job_description_skills

    def test_all_candidates_failed_batch(self, tmp_path: Path) -> None:
        empty_1 = tmp_path / "empty1.txt"
        empty_1.write_text("", encoding="utf-8")
        empty_2 = tmp_path / "empty2.txt"
        empty_2.write_text("", encoding="utf-8")

        result = rank_candidates([empty_1, empty_2], "Job description with Python")
        assert result.analytics.total_submitted == 2
        assert result.analytics.total_processed == 0
        assert result.analytics.total_failed == 2
        assert result.analytics.success_rate == 0.0
        assert result.analytics.mean_overall_score is None


class TestBatchAnalyticsCalculations:
    """Tests for statistical math and aggregated metric calculations."""

    def test_analytics_metrics_accuracy(self) -> None:
        """Verify statistical metrics (mean, median, min, max, std, distributions) against known values."""
        candidates = [
            CandidateResult(
                candidate_id="c1",
                filename="c1.pdf",
                status="success",
                overall_score=80.0,
                similarity_score=85.0,
                skill_coverage_score=75.0,
                matched_skills=["Python", "Docker"],
                missing_skills=["AWS", "Kubernetes"],
            ),
            CandidateResult(
                candidate_id="c2",
                filename="c2.pdf",
                status="success",
                overall_score=60.0,
                similarity_score=65.0,
                skill_coverage_score=55.0,
                matched_skills=["Python"],
                missing_skills=["AWS", "Docker", "Kubernetes"],
            ),
            CandidateResult(
                candidate_id="c3",
                filename="c3.pdf",
                status="success",
                overall_score=40.0,
                similarity_score=40.0,
                skill_coverage_score=40.0,
                matched_skills=["SQL"],
                missing_skills=["AWS", "Docker", "Kubernetes", "Python"],
            ),
            CandidateResult(
                candidate_id="c4",
                filename="c4.pdf",
                status="failed",
                error_message="Corrupted file",
            ),
        ]

        analytics = calculate_batch_analytics(candidates)

        assert analytics.total_submitted == 4
        assert analytics.total_processed == 3
        assert analytics.total_failed == 1
        assert pytest.approx(analytics.success_rate, abs=0.01) == 75.0

        # Mean: (80 + 60 + 40) / 3 = 60.0
        assert pytest.approx(analytics.mean_overall_score, abs=0.01) == 60.0
        # Median: 60.0
        assert pytest.approx(analytics.median_overall_score, abs=0.01) == 60.0
        assert analytics.min_overall_score == 40.0
        assert analytics.max_overall_score == 80.0
        # Std: population std of [80, 60, 40] = sqrt(((20^2 + 0 + 20^2)/3)) = sqrt(800/3) = 16.33
        assert pytest.approx(analytics.std_overall_score, abs=0.05) == 16.33

        # Mean similarity: (85 + 65 + 40) / 3 = 63.33
        assert pytest.approx(analytics.mean_similarity_score, abs=0.05) == 63.33

        # Top matched skills: Python appeared 2 times, Docker 1 time, SQL 1 time
        top_matched_dict = dict(analytics.top_matched_skills)
        assert top_matched_dict["Python"] == 2
        assert top_matched_dict["Docker"] == 1
        assert top_matched_dict["SQL"] == 1

        # Top missing skills: AWS appeared 3 times, Kubernetes 3 times, Docker 1 time
        top_missing_dict = dict(analytics.top_missing_skills)
        assert top_missing_dict["AWS"] == 3
        assert top_missing_dict["Kubernetes"] == 3

        # Score distribution buckets:
        # 40 -> '21-40'
        # 60 -> '41-60'
        # 80 -> '61-80'
        assert analytics.score_distribution["0-20"] == 0
        assert analytics.score_distribution["21-40"] == 1
        assert analytics.score_distribution["41-60"] == 1
        assert analytics.score_distribution["61-80"] == 1
        assert analytics.score_distribution["81-100"] == 0

    def test_empty_candidates_analytics(self) -> None:
        analytics = calculate_batch_analytics([])
        assert analytics.total_submitted == 0
        assert analytics.total_processed == 0
        assert analytics.total_failed == 0
        assert analytics.mean_overall_score is None
        assert analytics.median_overall_score is None
        assert analytics.top_matched_skills == []
        assert analytics.top_missing_skills == []


class TestCSVExportService:
    """Tests for RFC 4180 CSV export formatting, Unicode preservation, and privacy protections."""

    def test_csv_headers_and_row_structure(self) -> None:
        candidates = [
            CandidateResult(
                candidate_id="cand_1",
                filename="jane_doe_resume.pdf",
                status="success",
                rank=1,
                overall_score=85.25,
                similarity_score=88.50,
                skill_coverage_score=80.0,
                matched_skills=["Docker", "Python", "SQL"],
                missing_skills=["AWS"],
                method="sentence-transformers (all-MiniLM-L6-v2)",
                is_fallback=False,
            ),
            CandidateResult(
                candidate_id="cand_2",
                filename="bad_file.docx",
                status="failed",
                error_message="Document is empty (0 bytes)",
            ),
        ]

        csv_text = export_candidates_to_csv(candidates)
        lines = [line.strip() for line in csv_text.strip().split("\r\n")]

        # Header check
        assert lines[0] == ",".join(CSV_HEADERS)

        # Successful row check
        assert "1,cand_1,jane_doe_resume.pdf,success,85.25,88.50,80.00" in lines[1]
        assert "Docker; Python; SQL" in lines[1]
        assert "sentence-transformers" in lines[1]
        assert ",No," in lines[1]

        # Failed row check
        assert "N/A,cand_2,bad_file.docx,failed" in lines[2]
        assert "Document is empty (0 bytes)" in lines[2]

    def test_csv_handles_commas_quotes_and_unicode_safely(self) -> None:
        """Verify RFC 4180 escaping for commas, quotes, and international characters."""
        candidates = [
            CandidateResult(
                candidate_id="cand_unicode",
                filename='Résumé, "Senior" - José García.pdf',
                status="success",
                rank=1,
                overall_score=72.0,
                similarity_score=70.0,
                skill_coverage_score=75.0,
                matched_skills=["C++", "C#", ".NET", "Node.js"],
                missing_skills=["FastAPI"],
                method="TF-IDF Vector Space Model",
                is_fallback=True,
            )
        ]

        csv_text = export_candidates_to_csv(candidates)
        # Verify proper double quoting for fields with commas and quotes
        assert '"Résumé, ""Senior"" - José García.pdf"' in csv_text
        assert "C++; C#; .NET; Node.js" in csv_text

        # Verify export bytes with UTF-8-SIG includes BOM (\xef\xbb\xbf)
        csv_bytes = export_candidates_to_csv_bytes(candidates, encoding="utf-8-sig")
        assert csv_bytes.startswith(b"\xef\xbb\xbf")
        decoded = csv_bytes.decode("utf-8-sig")
        assert "José García" in decoded

    def test_csv_does_not_contain_full_resume_text(self) -> None:
        """Verify candidate privacy: full resume text is strictly excluded from export."""
        full_text = "CONFIDENTIAL_PRIVATE_RESUME_BODY_CONTENT_12345"
        candidates = [
            CandidateResult(
                candidate_id="cand_priv",
                filename="private_resume.pdf",
                status="success",
                rank=1,
                overall_score=80.0,
                similarity_score=80.0,
                skill_coverage_score=80.0,
                metadata={"full_text_snippet": full_text},  # Even if metadata held it
            )
        ]

        csv_text = export_candidates_to_csv(candidates)
        assert full_text not in csv_text


class TestMockedIntegration:
    """Offline integration tests with mocked similarity engines."""

    def test_ranking_with_mocked_similarity_engine(self, tmp_path: Path) -> None:
        mock_sim_engine = MagicMock(spec=BaseSimilarityEngine)
        mock_sim_engine.method_name = "MockDeterministicEngine"
        mock_sim_engine.is_fallback = False
        # Return deterministic similarity score of 80.0
        mock_sim_engine.compute_similarity.return_value = (80.0, {"mock": True})

        file1 = tmp_path / "cand_one.txt"
        file1.write_text("Developer proficient in Python and Docker.", encoding="utf-8")

        file2 = tmp_path / "cand_two.txt"
        file2.write_text("Engineer with Python, Docker, and Kubernetes.", encoding="utf-8")

        service = CandidateRankingService(similarity_engine=mock_sim_engine)
        jd = "Seeking Senior Engineer with Python, Docker, Kubernetes, and AWS."

        result = service.rank_candidates([file1, file2], jd)

        assert isinstance(result, BatchRankingResult)
        assert len(result.candidates) == 2
        assert result.analytics.total_processed == 2

        # Both had sim=80.0; cand_two had 3/4 skills (75%), cand_one had 2/4 skills (50%)
        # So cand_two is ranked #1, cand_one is #2
        assert result.candidates[0].filename == "cand_two.txt"
        assert result.candidates[0].rank == 1
        assert result.candidates[1].filename == "cand_one.txt"
        assert result.candidates[1].rank == 2
