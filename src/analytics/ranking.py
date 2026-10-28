"""
Candidate Ranking Service for ResumeMatch AI.
Course: CSE472 Deep Learning for NLP
Milestone: Day 6 Candidate Ranking & Analytics

Coordinates batch document ingestion, parsing, preprocessing, semantic matching,
deterministic tie-breaking, failure isolation, and analytics generation.
"""

import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from config.settings import settings
from src.analytics.analytics import calculate_batch_analytics
from src.analytics.models import BatchRankingResult, CandidateResult
from src.matching.engine import SemanticMatchingEngine
from src.matching.models import BaseSimilarityEngine
from src.nlp.feature_extractor import extract_features
from src.parsers.exceptions import ParserError
from src.parsers.service import DocumentParser
from src.preprocessing.normalizer import normalize_text
from src.utils.logger import get_logger

logger = get_logger("resumematch.ranking")

ResumeInput = Union[str, Path, Tuple[str, Union[bytes, bytearray]], Any]


class CandidateRankingService:
    """
    Modular batch candidate ranking service.

    Ingests multiple candidate resumes (PDF, DOCX, TXT), validates document integrity,
    normalizes textual streams, executes semantic and skill alignment, applies
    deterministic tie-breaking, and isolates individual file errors from batch failures.
    """

    def __init__(
        self,
        parser: Optional[DocumentParser] = None,
        matching_engine: Optional[SemanticMatchingEngine] = None,
        similarity_engine: Optional[BaseSimilarityEngine] = None,
        semantic_weight: Optional[float] = None,
        skill_weight: Optional[float] = None,
        matching_method: Optional[str] = None,
    ) -> None:
        """
        Initialize CandidateRankingService.

        Args:
            parser: DocumentParser instance, or None to initialize with defaults.
            matching_engine: SemanticMatchingEngine instance, or None to initialize with defaults.
            similarity_engine: Optional BaseSimilarityEngine implementation.
            semantic_weight: Optional weight for semantic similarity (default: 0.6).
            skill_weight: Optional weight for skill coverage (default: 0.4).
            matching_method: Optional matching strategy ('auto', 'sentence-transformers', 'tfidf').
        """
        self.parser = parser or DocumentParser()
        self.matching_engine = matching_engine or SemanticMatchingEngine(
            similarity_engine=similarity_engine,
            semantic_weight=semantic_weight,
            skill_weight=skill_weight,
            matching_method=matching_method,
        )

    def _resolve_candidate_input(
        self,
        item: ResumeInput,
        index: int,
        temp_dir: Path,
    ) -> Tuple[str, str, Path]:
        """
        Normalize diverse candidate inputs into a filename, candidate_id, and resolved filesystem Path.

        Supports:
            - str / Path file paths on disk.
            - Tuples of (filename, bytes).
            - Streamlit UploadedFile objects (having .name and .getbuffer() or .read()).

        Returns:
            Tuple of (filename, candidate_id, file_path).
        """
        if isinstance(item, (str, Path)):
            resolved_path = Path(item).resolve()
            filename = resolved_path.name
            candidate_id = f"cand_{index + 1}_{resolved_path.stem}"
            return filename, candidate_id, resolved_path

        if isinstance(item, tuple) and len(item) == 2:
            filename, raw_bytes = item
            if not isinstance(filename, str):
                filename = str(filename)
            candidate_id = f"cand_{index + 1}_{Path(filename).stem}"
            temp_path = temp_dir / filename
            temp_path.write_bytes(bytes(raw_bytes))
            return filename, candidate_id, temp_path

        # Streamlit UploadedFile or file-like buffer
        if hasattr(item, "name") and (hasattr(item, "getbuffer") or hasattr(item, "read")):
            filename = str(item.name)
            candidate_id = f"cand_{index + 1}_{Path(filename).stem}"
            content = item.getbuffer() if hasattr(item, "getbuffer") else item.read()
            temp_path = temp_dir / filename
            temp_path.write_bytes(bytes(content))
            return filename, candidate_id, temp_path

        # Fallback for unexpected object types
        filename = f"document_{index + 1}.bin"
        candidate_id = f"cand_{index + 1}_unknown"
        temp_path = temp_dir / filename
        temp_path.write_text(str(item), encoding="utf-8")
        return filename, candidate_id, temp_path

    def rank_candidates(
        self,
        resumes: List[ResumeInput],
        job_description_text: str,
    ) -> BatchRankingResult:
        """
        Rank a batch of candidate resumes against a target job description.

        Args:
            resumes: List of candidate inputs (paths, (filename, bytes) tuples, or uploaded files).
            job_description_text: Target job description text.

        Returns:
            BatchRankingResult containing sorted candidates and aggregate analytics.

        Raises:
            TypeError: If job_description_text is not a string or resumes is not a list.
            ValueError: If job_description_text is empty or whitespace-only.
        """
        if not isinstance(job_description_text, str):
            raise TypeError(
                f"Job description must be a string, got {type(job_description_text).__name__}"
            )

        if not isinstance(resumes, list):
            raise TypeError(
                f"Resumes must be a list of documents, got {type(resumes).__name__}"
            )

        clean_jd_raw = job_description_text.strip()
        if not clean_jd_raw:
            raise ValueError("Target job description cannot be empty or whitespace-only.")

        # Normalize target job description once for the entire batch
        normalized_jd = normalize_text(clean_jd_raw)
        jd_features = extract_features(normalized_jd)
        jd_skills = jd_features.get_skill_names()

        if not resumes:
            empty_analytics = calculate_batch_analytics([], jd_skills=jd_skills)
            return BatchRankingResult(
                candidates=[],
                analytics=empty_analytics,
                job_description_skills=jd_skills,
                metadata={"total_submitted": 0},
            )

        candidate_results: List[CandidateResult] = []

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)

            for idx, resume_item in enumerate(resumes):
                filename = f"resume_{idx + 1}"
                candidate_id = f"cand_{idx + 1}"

                try:
                    filename, candidate_id, file_path = self._resolve_candidate_input(
                        resume_item, idx, temp_dir
                    )
                except Exception as exc:
                    logger.warning("Failed to resolve candidate input #%d: %s", idx + 1, exc)
                    candidate_results.append(
                        CandidateResult(
                            candidate_id=candidate_id,
                            filename=filename,
                            status="failed",
                            error_message=f"Input resolution error: {exc}",
                        )
                    )
                    continue

                # 1. Parse Document (with error containment)
                try:
                    parsed_doc = self.parser.parse(file_path)
                except ParserError as p_err:
                    logger.warning("Document parsing failed for '%s': %s", filename, p_err)
                    candidate_results.append(
                        CandidateResult(
                            candidate_id=candidate_id,
                            filename=filename,
                            status="failed",
                            error_message=str(p_err),
                        )
                    )
                    continue
                except Exception as gen_err:
                    logger.warning("Unexpected extraction error for '%s': %s", filename, gen_err)
                    candidate_results.append(
                        CandidateResult(
                            candidate_id=candidate_id,
                            filename=filename,
                            status="failed",
                            error_message=f"Extraction failure: {gen_err}",
                        )
                    )
                    continue

                # 2. Normalize Text
                try:
                    normalized_resume = normalize_text(parsed_doc.text)
                except Exception as norm_err:
                    logger.warning("Normalization error for '%s': %s", filename, norm_err)
                    candidate_results.append(
                        CandidateResult(
                            candidate_id=candidate_id,
                            filename=filename,
                            status="failed",
                            error_message=f"Normalization failure: {norm_err}",
                        )
                    )
                    continue

                if not normalized_resume.strip():
                    logger.warning("Document '%s' contains no extractable text after normalization", filename)
                    candidate_results.append(
                        CandidateResult(
                            candidate_id=candidate_id,
                            filename=filename,
                            status="failed",
                            error_message="Document contains no readable text content after normalization.",
                        )
                    )
                    continue

                # 3. Match against Job Description
                try:
                    match_res = self.matching_engine.match(normalized_resume, normalized_jd)
                except Exception as match_err:
                    logger.warning("Semantic matching error for '%s': %s", filename, match_err)
                    candidate_results.append(
                        CandidateResult(
                            candidate_id=candidate_id,
                            filename=filename,
                            status="failed",
                            error_message=f"Matching failure: {match_err}",
                        )
                    )
                    continue

                # 4. Record Successful Candidate Result
                candidate_results.append(
                    CandidateResult(
                        candidate_id=candidate_id,
                        filename=filename,
                        status="success",
                        overall_score=match_res.overall_score,
                        similarity_score=match_res.similarity_score,
                        skill_coverage_score=match_res.skill_coverage_score,
                        matched_skills=match_res.matched_skills,
                        missing_skills=match_res.missing_skills,
                        resume_skills=match_res.resume_skills,
                        method=match_res.method,
                        is_fallback=match_res.is_fallback,
                        match_result=match_res,
                        metadata={
                            "file_size_bytes": parsed_doc.file_size_bytes,
                            "character_count": parsed_doc.character_count,
                            "word_count": parsed_doc.word_count,
                            "page_count": parsed_doc.page_count,
                            "file_type": parsed_doc.file_type,
                        },
                    )
                )

        # 5. Deterministic Tie-Breaking and Sorting
        # Successful candidates sorted by:
        #   1. overall_score (descending)
        #   2. similarity_score (descending)
        #   3. skill_coverage_score (descending)
        #   4. filename lowercased (ascending)
        #   5. candidate_id (ascending)
        successful_candidates = [c for c in candidate_results if c.status == "success"]
        failed_candidates = [c for c in candidate_results if c.status == "failed"]

        successful_candidates.sort(
            key=lambda c: (
                -(c.overall_score if c.overall_score is not None else 0.0),
                -(c.similarity_score if c.similarity_score is not None else 0.0),
                -(c.skill_coverage_score if c.skill_coverage_score is not None else 0.0),
                c.filename.lower(),
                c.candidate_id,
            )
        )

        # Assign 1-indexed ranks to successfully evaluated candidates
        for rank_idx, candidate in enumerate(successful_candidates, start=1):
            candidate.rank = rank_idx

        # Failed candidates remain unranked (rank=None) and placed at the tail
        for candidate in failed_candidates:
            candidate.rank = None

        ranked_all = successful_candidates + failed_candidates

        # 6. Aggregate Batch Analytics
        analytics = calculate_batch_analytics(ranked_all, jd_skills=jd_skills)

        logger.info(
            "Batch ranking complete: %d submitted, %d processed, %d failed",
            len(ranked_all),
            analytics.total_processed,
            analytics.total_failed,
        )

        return BatchRankingResult(
            candidates=ranked_all,
            analytics=analytics,
            job_description_skills=jd_skills,
            metadata={
                "total_submitted": len(ranked_all),
                "semantic_weight": self.matching_engine.semantic_weight,
                "skill_weight": self.matching_engine.skill_weight,
            },
        )


def rank_candidates(
    resumes: List[ResumeInput],
    job_description_text: str,
    parser: Optional[DocumentParser] = None,
    matching_engine: Optional[SemanticMatchingEngine] = None,
    similarity_engine: Optional[BaseSimilarityEngine] = None,
    semantic_weight: Optional[float] = None,
    skill_weight: Optional[float] = None,
    matching_method: Optional[str] = None,
) -> BatchRankingResult:
    """
    Convenience function executing candidate ranking and batch analytics.

    Args:
        resumes: List of candidate resumes (paths, tuples, or upload objects).
        job_description_text: Target job description text.
        parser: Optional DocumentParser instance.
        matching_engine: Optional SemanticMatchingEngine instance.
        similarity_engine: Optional BaseSimilarityEngine instance.
        semantic_weight: Optional weight for semantic similarity (0.0-1.0).
        skill_weight: Optional weight for skill coverage (0.0-1.0).
        matching_method: Optional matching strategy ('auto', 'sentence-transformers', 'tfidf').

    Returns:
        BatchRankingResult containing ranked candidates and batch analytics.
    """
    service = CandidateRankingService(
        parser=parser,
        matching_engine=matching_engine,
        similarity_engine=similarity_engine,
        semantic_weight=semantic_weight,
        skill_weight=skill_weight,
        matching_method=matching_method,
    )
    return service.rank_candidates(resumes, job_description_text)
