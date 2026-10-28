"""
Data models and schemas for candidate ranking, batch evaluation, and analytics.
Course: CSE472 Deep Learning for NLP
Milestone: Day 6 Candidate Ranking & Analytics
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.matching.schemas import MatchResult


@dataclass
class CandidateResult:
    """
    Structured outcome of an individual candidate resume evaluation against a target job description.

    Attributes:
        candidate_id: Unique candidate or document identifier.
        filename: Original filename of the submitted resume.
        status: Processing outcome ('success' or 'failed').
        rank: 1-indexed relative rank among successfully processed candidates (None if failed).
        overall_score: Configurable heuristic match score (0.0 to 100.0). None if failed.
        similarity_score: Vector cosine similarity on a 0.0-100.0 display scale. None if failed.
        skill_coverage_score: Percentage of job description skills matched (0.0 to 100.0). None if failed.
        matched_skills: Canonical skills detected in both JD and candidate resume.
        missing_skills: Canonical skills requested in JD but not identified in candidate resume.
        resume_skills: All canonical technical skills identified in the candidate resume.
        method: Identifier of the matching model/backend used.
        is_fallback: True if a lexical/TF-IDF fallback model was used.
        error_message: Detailed diagnostic error message if status == 'failed'.
        match_result: Full MatchResult instance for detailed inspection (optional).
        metadata: Format metadata, execution diagnostics, or token counts.
    """

    candidate_id: str
    filename: str
    status: str
    rank: Optional[int] = None
    overall_score: Optional[float] = None
    similarity_score: Optional[float] = None
    skill_coverage_score: Optional[float] = None
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    resume_skills: List[str] = field(default_factory=list)
    method: Optional[str] = None
    is_fallback: Optional[bool] = None
    error_message: Optional[str] = None
    match_result: Optional[MatchResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize candidate result to a dictionary representation."""
        data = asdict(self)
        if self.match_result:
            data["match_result"] = self.match_result.to_dict()
        return data


@dataclass
class BatchAnalytics:
    """
    Aggregated statistical summary and skill analytics for a batch of candidate resumes.

    Attributes:
        total_submitted: Total count of resume documents submitted in the batch.
        total_processed: Count of documents parsed, preprocessed, and matched successfully.
        total_failed: Count of documents that could not be processed due to errors.
        success_rate: Percentage of submitted resumes successfully evaluated (0.0 to 100.0).
        mean_overall_score: Arithmetic mean of overall heuristic scores for successful candidates.
        median_overall_score: Median of overall heuristic scores for successful candidates.
        min_overall_score: Minimum overall score in the batch.
        max_overall_score: Maximum overall score in the batch.
        std_overall_score: Standard deviation of overall scores in the batch.
        mean_similarity_score: Average semantic vector similarity score.
        mean_skill_coverage_score: Average skill coverage percentage.
        top_matched_skills: Most frequently detected matching skills with frequency counts.
        top_missing_skills: Most frequently missing job description skills with frequency counts.
        score_distribution: Frequency counts of overall scores partitioned into standard buckets.
    """

    total_submitted: int
    total_processed: int
    total_failed: int
    success_rate: float
    mean_overall_score: Optional[float] = None
    median_overall_score: Optional[float] = None
    min_overall_score: Optional[float] = None
    max_overall_score: Optional[float] = None
    std_overall_score: Optional[float] = None
    mean_similarity_score: Optional[float] = None
    mean_skill_coverage_score: Optional[float] = None
    top_matched_skills: List[Tuple[str, int]] = field(default_factory=list)
    top_missing_skills: List[Tuple[str, int]] = field(default_factory=list)
    score_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize batch analytics summary to a dictionary."""
        return asdict(self)


@dataclass
class BatchRankingResult:
    """
    Complete response payload for a batch candidate ranking and analytics evaluation.

    Attributes:
        candidates: List of CandidateResult instances, sorted by overall score descending.
        analytics: Aggregated BatchAnalytics summary.
        job_description_skills: List of canonical technical skills detected in the target JD.
        timestamp: ISO 8601 UTC timestamp of when the evaluation was executed.
        metadata: Technical configuration or batch provenance metadata.
    """

    candidates: List[CandidateResult]
    analytics: BatchAnalytics
    job_description_skills: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire batch ranking result to a dictionary."""
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "analytics": self.analytics.to_dict(),
            "job_description_skills": self.job_description_skills,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
