"""
Data models and output schemas for the ResumeMatch AI matching engine.
Course: CSE472 Deep Learning for NLP
Milestone: Day 5 Semantic Matching Engine
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class MatchResult:
    """
    Structured outcome of a semantic and skill matching evaluation between
    a candidate resume and a target job description.

    Attributes:
        similarity_score: Cosine similarity mapped to a 0.0-100.0 display scale
            (max(0.0, min(100.0, cosine * 100.0))). NOTE: This score is a scaled vector
            similarity metric, NOT a probability, statistical confidence score, or calibrated
            percentage of job suitability. When using SentenceTransformers, it reflects
            dense contextual embedding vector similarity; when using TF-IDF, it reflects
            lexical term-overlap vector similarity.
        skill_coverage_score: Percentage of identified job description taxonomy skills
            matched in the resume (0.0-100.0).
        overall_score: Configurable heuristic index (0.0-100.0) combining semantic similarity
            and skill coverage (default formula: 0.6 * similarity_score + 0.4 * skill_coverage_score).
            NOTE: This is an exploratory matching heuristic, NOT a validated hiring prediction,
            candidate qualification grade, or automated employment decision metric.
        method: Human-readable identifier of the embedding/matching model used.
        is_fallback: True if a lexical/TF-IDF fallback model was used instead of neural embeddings.
        matched_skills: Alphabetically sorted list of skills present in both JD and resume.
        missing_skills: Alphabetically sorted list of skills requested in JD but not found in resume.
        resume_skills: All canonical skills detected in the resume text.
        job_description_skills: All canonical skills detected in the job description text.
        explanation: Objective, evidence-based assessment summary in non-judgmental language.
        metadata: Additional technical diagnostics (e.g. raw cosine score, applied weights, fallback details).
    """

    similarity_score: float
    skill_coverage_score: float
    overall_score: float
    method: str
    is_fallback: bool
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    resume_skills: List[str] = field(default_factory=list)
    job_description_skills: List[str] = field(default_factory=list)
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize match results to a standard dictionary format."""
        return asdict(self)
