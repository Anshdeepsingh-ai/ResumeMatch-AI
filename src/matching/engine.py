"""
Semantic Matching Engine for ResumeMatch AI.
Course: CSE472 Deep Learning for NLP
Milestone: Day 5 Semantic Matching Engine

Coordinates semantic vector similarity, skill extraction alignment,
weighted composite scoring, and evidence-based non-judgmental explanations.
"""

from typing import Any, Dict, List, Optional

from config.settings import settings
from src.matching.models import BaseSimilarityEngine, get_similarity_engine
from src.matching.schemas import MatchResult
from src.nlp.feature_extractor import extract_features
from src.utils.logger import get_logger

logger = get_logger("resumematch.matching")


def generate_explanation(
    similarity_score: float,
    skill_coverage_score: float,
    overall_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
    jd_skills: List[str],
    resume_skills: List[str],
    method_name: str,
    is_fallback: bool,
) -> str:
    """
    Generate an objective, evidence-based assessment summary in non-judgmental language.

    Adheres to ethical recruitment principles:
    - Clearly distinguishes dense embedding similarity from lexical TF-IDF overlap.
    - Labels scores as display-scale similarity metrics, not probabilities or hiring predictions.
    - Does not claim missing keywords prove candidate incompetence or disqualification.
    """
    parts: List[str] = []

    # 1. Semantic alignment observation
    if similarity_score >= 75.0:
        align_desc = "strong contextual alignment"
    elif similarity_score >= 50.0:
        align_desc = "moderate contextual alignment"
    elif similarity_score >= 25.0:
        align_desc = "partial contextual alignment"
    else:
        align_desc = "limited contextual alignment"

    metric_type = "lexical overlap" if is_fallback else "dense embedding similarity"
    parts.append(
        f"Semantic comparison yielded an alignment score of {similarity_score:.1f}/100 ({align_desc}), "
        f"reflecting {metric_type} via cosine similarity."
    )

    # 2. Skill taxonomy comparison
    if jd_skills:
        n_matched = len(matched_skills)
        n_total = len(jd_skills)
        parts.append(
            f"Technical skill analysis identified {n_matched} of {n_total} job description skills "
            f"({skill_coverage_score:.1f}% coverage)."
        )
        if matched_skills:
            parts.append(f"Directly matched skills: {', '.join(matched_skills)}.")
        if missing_skills:
            parts.append(
                f"Job description skills not explicitly identified in the resume text: {', '.join(missing_skills)}."
            )
    else:
        parts.append(
            "The job description did not specify canonical technical skills from the standard taxonomy."
        )

    # 3. Composite score (heuristic labeling)
    parts.append(
        f"Composite overall score is {overall_score:.1f}/100 (a configurable heuristic index combining "
        f"{similarity_score:.1f} similarity and {skill_coverage_score:.1f}% skill coverage)."
    )

    # 4. Method transparency
    if is_fallback:
        parts.append(
            "Note: Lexical TF-IDF vector space modeling was used as an operational fallback."
        )

    # 5. Ethical disclaimer
    parts.append(
        "Disclaimer: Scores represent scaled cosine similarity (0–100 display scale) and exploratory heuristics, "
        "not probabilities, calibrated suitability percentages, or validated hiring predictions. "
        "Textual matching reflects documented keyword correspondence; absence of specific mentions does not imply "
        "a lack of candidate qualification."
    )

    return " ".join(parts)


class SemanticMatchingEngine:
    """
    Core matching service evaluating alignment between resumes and job descriptions.

    Integrates deep semantic embeddings, pure-NumPy TF-IDF fallbacks,
    skill taxonomy extraction, and configurable composite scoring.
    """

    def __init__(
        self,
        similarity_engine: Optional[BaseSimilarityEngine] = None,
        semantic_weight: Optional[float] = None,
        skill_weight: Optional[float] = None,
        matching_method: Optional[str] = None,
    ) -> None:
        """
        Initialize the matching engine.

        Args:
            similarity_engine: Concrete BaseSimilarityEngine implementation, or None to auto-detect.
            semantic_weight: Weight assigned to semantic similarity (default: 0.6).
            skill_weight: Weight assigned to skill coverage (default: 0.4).
            matching_method: Optional method selection strategy ('auto', 'sentence-transformers', 'tfidf').
        """
        method = matching_method or getattr(settings, "MATCHING_METHOD", "auto")
        self.similarity_engine = similarity_engine or get_similarity_engine(method)

        raw_sem = (
            semantic_weight
            if semantic_weight is not None
            else getattr(settings, "SEMANTIC_WEIGHT", 0.6)
        )
        raw_skill = (
            skill_weight
            if skill_weight is not None
            else getattr(settings, "SKILL_WEIGHT", 0.4)
        )

        if raw_sem < 0.0 or raw_skill < 0.0:
            raise ValueError(
                f"Weights must be non-negative: semantic={raw_sem}, skill={raw_skill}"
            )

        total_weight = raw_sem + raw_skill
        if abs(total_weight - 1.0) > 1e-4:
            raise ValueError(
                f"Weights must sum to 1.0 (got {total_weight:.4f} from semantic={raw_sem}, skill={raw_skill})"
            )

        self.semantic_weight = float(raw_sem)
        self.skill_weight = float(raw_skill)

    def match(
        self,
        resume_text: str,
        job_description_text: str,
    ) -> MatchResult:
        """
        Perform complete semantic and skill alignment evaluation.

        Args:
            resume_text: Raw or preprocessed candidate resume text.
            job_description_text: Raw or preprocessed target job description text.

        Returns:
            MatchResult containing similarity score, skill coverage, overall score,
            matched/missing skills, and non-judgmental explanation.

        Raises:
            TypeError: If either input is not a string.
        """
        if not isinstance(resume_text, str) or not isinstance(job_description_text, str):
            raise TypeError(
                f"Expected string inputs, got resume={type(resume_text).__name__}, "
                f"job_description={type(job_description_text).__name__}"
            )

        clean_resume = resume_text.strip()
        clean_jd = job_description_text.strip()

        # Handle empty or whitespace-only inputs
        if not clean_resume or not clean_jd:
            return MatchResult(
                similarity_score=0.0,
                skill_coverage_score=0.0,
                overall_score=0.0,
                method=self.similarity_engine.method_name,
                is_fallback=self.similarity_engine.is_fallback,
                matched_skills=[],
                missing_skills=[],
                resume_skills=[],
                job_description_skills=[],
                explanation=(
                    "Evaluation could not be performed because one or both input documents "
                    "(resume or job description) are empty or contain no text content."
                ),
                metadata={
                    "reason": "empty_document_input",
                    "resume_empty": not bool(clean_resume),
                    "job_description_empty": not bool(clean_jd),
                },
            )

        # 1. NLP Feature & Skill Extraction
        resume_features = extract_features(clean_resume)
        jd_features = extract_features(clean_jd)

        resume_skills = resume_features.get_skill_names()
        jd_skills = jd_features.get_skill_names()

        matched_skills = sorted(list(set(jd_skills) & set(resume_skills)))
        missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))

        # 2. Skill Coverage Calculation (0.0 to 100.0)
        if jd_skills:
            skill_coverage = round((len(matched_skills) / len(jd_skills)) * 100.0, 2)
        else:
            skill_coverage = 100.0

        # 3. Semantic Similarity Calculation (0.0 to 100.0)
        sim_score, sim_meta = self.similarity_engine.compute_similarity(
            clean_resume, clean_jd
        )
        sim_score = round(max(0.0, min(100.0, sim_score)), 2)

        # 4. Configurable Composite Overall Score
        overall = round(
            (self.semantic_weight * sim_score) + (self.skill_weight * skill_coverage),
            2,
        )
        overall = max(0.0, min(100.0, overall))

        # Accurately reflect method and fallback status from engine/metadata
        effective_method = sim_meta.get("method_name", self.similarity_engine.method_name)
        effective_fallback = sim_meta.get("is_fallback", self.similarity_engine.is_fallback)

        # 5. Objective Evidence-Based Explanation
        explanation = generate_explanation(
            similarity_score=sim_score,
            skill_coverage_score=skill_coverage,
            overall_score=overall,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            jd_skills=jd_skills,
            resume_skills=resume_skills,
            method_name=effective_method,
            is_fallback=effective_fallback,
        )

        metadata: Dict[str, Any] = {
            "semantic_weight": self.semantic_weight,
            "skill_weight": self.skill_weight,
            "similarity_engine_metadata": sim_meta,
            "resume_token_count": resume_features.statistics.token_count,
            "jd_token_count": jd_features.statistics.token_count,
        }

        return MatchResult(
            similarity_score=sim_score,
            skill_coverage_score=skill_coverage,
            overall_score=overall,
            method=effective_method,
            is_fallback=effective_fallback,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            resume_skills=resume_skills,
            job_description_skills=jd_skills,
            explanation=explanation,
            metadata=metadata,
        )


def match_resume_to_job(
    resume_text: str,
    job_description_text: str,
    similarity_engine: Optional[BaseSimilarityEngine] = None,
    semantic_weight: Optional[float] = None,
    skill_weight: Optional[float] = None,
    matching_method: Optional[str] = None,
) -> MatchResult:
    """
    Convenience function evaluating match between a resume and job description.

    Args:
        resume_text: Resume text string.
        job_description_text: Job description text string.
        similarity_engine: Optional similarity engine instance.
        semantic_weight: Optional weight for semantic similarity (0.0-1.0).
        skill_weight: Optional weight for skill coverage (0.0-1.0).
        matching_method: Optional method strategy ('auto', 'sentence-transformers', 'tfidf').

    Returns:
        Structured MatchResult instance.
    """
    engine = SemanticMatchingEngine(
        similarity_engine=similarity_engine,
        semantic_weight=semantic_weight,
        skill_weight=skill_weight,
        matching_method=matching_method,
    )
    return engine.match(resume_text, job_description_text)
